"""명령 해석 모듈 단위 시험. Claude API 대신 정해진 응답을 돌려주는 가짜 클라이언트를 쓴다."""
import json
import threading
from types import SimpleNamespace as NS

import pytest

from app.agent import offline
from app.agent.agent import CommandAgent
from app.agent.config import AgentConfig
from app.agent.eval import grade
from app.agent.robot import MockRobot
from app.agent.runner import VoiceCommandRunner
from app.agent.tools import TOOLS_BY_NAME, build_tools, validate_input


# ---- 가짜 Claude ----------------------------------------------------------
def text(t):
    return NS(type="text", text=t)


def use(tool, i=0, **inp):
    return NS(type="tool_use", id=f"tu_{tool}_{i}", name=tool, input=inp)


def resp(*blocks, stop=None):
    if stop is None:
        stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    return NS(content=list(blocks), stop_reason=stop, usage=None)


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        # 요청 당시의 messages를 복사해 둔다 (이후 append로 바뀌지 않게)
        self.requests.append({**kw, "messages": [dict(m) for m in kw["messages"]]})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make(*responses, cfg=None, robot=None):
    client = FakeClient(*responses)
    robot = robot or MockRobot()
    events = []
    agent = CommandAgent(robot, cfg or AgentConfig(), client=client, on_event=events.append)
    return agent, client, robot, events


# ---- 기본 흐름 -------------------------------------------------------------
def test_hand_over_flow_runs_tools_and_returns_text():
    agent, client, robot, _ = make(
        resp(use("find_objects", kind="screwdriver")),
        resp(use("hand_over", target_id="obj_4")),
        resp(text("드라이버 드렸습니다.")),
    )
    r = agent.handle("드라이버 줘")
    assert r.status == "done" and r.text == "드라이버 드렸습니다."
    assert [c.name for c in r.tool_calls] == ["find_objects", "hand_over"]
    assert [t for t, _ in robot.log] == ["find_objects", "hand_over"]
    assert "obj_4" not in robot.objects
    # 도구 결과가 LLM에 전달됐는지
    last_user = client.requests[1]["messages"][-1]
    assert last_user["role"] == "user" and last_user["content"][0]["type"] == "tool_result"
    found = json.loads(last_user["content"][0]["content"])
    assert found["count"] == 1 and found["objects"][0]["id"] == "obj_4"


def test_request_shape():
    agent, client, _, _ = make(resp(text("네.")))
    agent.handle("안녕")
    kw = client.requests[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"] == {"effort": "low"}
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert "툴메이트" in kw["system"][0]["text"]
    assert all(t["strict"] and t["input_schema"]["additionalProperties"] is False for t in kw["tools"])
    assert kw["messages"] == [{"role": "user", "content": [{"type": "text", "text": "안녕"}]}]


def test_low_confidence_is_not_sent():
    agent, client, _, _ = make()
    r = agent.handle("빨간거죠", confidence=0.45)
    assert r.status == "rejected" and client.requests == []


# ---- 되묻기 -----------------------------------------------------------------
def test_clarification_waits_for_next_utterance():
    agent, client, robot, _ = make(
        resp(use("find_objects", color="red", kind="block")),
        resp(use("ask_clarification", question="어느 빨간 블록이요?", options=["왼쪽", "오른쪽"])),
        resp(use("hand_over", target_id="obj_1")),
        resp(text("왼쪽 블록 드렸습니다.")),
    )
    r = agent.handle("빨간 블록 줘")
    assert r.status == "asked" and r.options == ["왼쪽", "오른쪽"]
    assert agent.waiting_answer
    assert [t for t, _ in robot.log] == ["find_objects"]   # 아직 집지 않음

    r = agent.handle("왼쪽")
    assert r.status == "done" and "obj_1" not in robot.objects
    # 답은 되묻기 도구 결과와 같은 user 메시지에 이어 붙는다
    msg = client.requests[2]["messages"][-1]
    assert [b["type"] for b in msg["content"]] == ["tool_result", "text"]
    assert msg["content"][1]["text"] == "왼쪽"
    roles = [m["role"] for m in client.requests[2]["messages"]]
    assert all(a != b for a, b in zip(roles, roles[1:]))   # user/assistant 번갈아


# ---- 안전, 한도 ------------------------------------------------------------
def test_tool_call_limit():
    many = [resp(use("gesture", i, name="nod")) for i in range(6)]
    agent, _, robot, _ = make(*many, cfg=AgentConfig(max_tool_calls=3))
    r = agent.handle("계속 끄덕여")
    assert r.status == "error"
    assert len(robot.log) == 3


def test_extra_calls_in_one_response_beyond_limit_are_not_run():
    agent, _, robot, _ = make(
        resp(*[use("gesture", i, name="nod") for i in range(4)]),
        cfg=AgentConfig(max_tool_calls=2),
    )
    r = agent.handle("끄덕여")
    assert len(robot.log) == 2 and r.status == "error"
    assert [c.is_error for c in r.tool_calls] == [False, False, True, True]


def test_cancel_skips_remaining_tools():
    robot = MockRobot()
    agent, _, _, _ = make(
        resp(use("find_objects"), use("tidy_up")),
        robot=robot,
    )
    orig = robot.run

    def run_then_stop(tool, args):
        out = orig(tool, args)
        agent.cancel()          # 첫 도구 실행 중 정지어가 들렸다고 가정
        return out

    robot.run = run_then_stop
    r = agent.handle("다 정리해")
    assert r.status == "cancelled"
    assert [t for t, _ in robot.log] == ["find_objects"]


def test_invalid_args_are_not_executed():
    agent, client, robot, _ = make(
        resp(use("hand_over", target=3)),
        resp(text("죄송합니다.")),
    )
    agent.handle("줘")
    assert robot.log == []
    res = client.requests[1]["messages"][-1]["content"][0]
    assert res["is_error"] is True and "잘못된 인자" in res["content"]


def test_tool_not_in_profile_is_refused():
    robot = MockRobot(capabilities={"gripper"})        # 카메라 없음
    agent, client, _, _ = make(resp(use("find_objects")), resp(text("카메라가 없어요.")), robot=robot)
    agent.handle("뭐 있어?")
    names = [t["name"] for t in client.requests[0]["tools"]]
    assert "find_objects" not in names and "pick_from_slot" in names
    assert robot.log == []


def test_build_tools_by_capability():
    full = {t.name for t in build_tools({"gripper", "vision.top"})}
    assert {"find_objects", "tidy_up", "hand_over"} <= full and "pick_from_slot" not in full
    no_grip = {t.name for t in build_tools({"vision.top"})}
    assert "push" in no_grip and "hand_over" not in no_grip
    bare = {t.name for t in build_tools(set())}
    assert bare == {"gesture", "go_home", "ask_clarification", "set_persona"}


def test_validate_input():
    spec = TOOLS_BY_NAME["gesture"]
    assert validate_input(spec, {"name": "nod"}) is None
    assert validate_input(spec, {"name": "dance"})
    assert validate_input(spec, {})
    assert validate_input(spec, {"name": "nod", "x": 1})


# ---- 페르소나 -----------------------------------------------------------------
def test_set_persona():
    agent, client, robot, _ = make(resp(use("set_persona", name="집사")), resp(text("분부대로 하겠습니다.")))
    agent.handle("집사로 바꿔줘")
    assert agent.persona == "집사" and robot.log == []
    res = client.requests[1]["messages"][-1]["content"][0]["content"]
    assert "집사" in res


# ---- 오류, 오프라인 ---------------------------------------------------------
def test_missing_credentials_falls_back_to_offline():
    err = TypeError("Could not resolve authentication method.")
    agent, _, robot, _ = make(err)
    r = agent.handle("제자리로 가")
    assert r.status == "offline" and robot.log == [("go_home", {})]
    assert agent.messages == []       # 실패한 요청은 기록에서 지움


def test_network_error_after_robot_moved_keeps_history():
    import anthropic

    agent, _, robot, _ = make(
        resp(use("go_home")),
        anthropic.APIConnectionError(request=None),
    )
    r = agent.handle("제자리로")
    assert r.status == "error" and robot.log == [("go_home", {})]   # 두 번 움직이지 않음
    assert agent.messages[-1]["role"] == "user"


def test_refusal_is_rejected_and_rolled_back():
    agent, _, _, _ = make(resp(text(""), stop="refusal"))
    r = agent.handle("...")
    assert r.status == "rejected" and agent.messages == []


def test_offline_mode_never_calls_llm():
    agent, client, robot, _ = make(cfg=AgentConfig(offline=True))
    assert agent.handle("인사해").status == "offline"
    assert agent.handle("2번 칸 거 줘").tool_calls[0].args == {"slot": 2, "place": "사용자"}
    assert agent.handle("커피 타 줘").text == offline.HELP
    assert client.requests == []


@pytest.mark.parametrize("text,expected", [
    ("제자리로 가", ("go_home", {})),
    ("대기 자세", ("go_home", {})),
    ("인사해 봐", ("gesture", {"name": "greet"})),
    ("고개 저어", ("gesture", {"name": "shake_head"})),
    ("1번 칸 물건 3번 칸으로 가져다 줘", ("pick_from_slot", {"slot": 1, "place": "3번 칸"})),
    ("세 번 칸 거 건네줘", ("pick_from_slot", {"slot": 3, "place": "사용자"})),
    ("집사로 바꿔", ("set_persona", {"name": "집사"})),
    ("고양이로 바꿔줘", ("set_persona", {"name": "겁 많은 고양이"})),
    ("츤데레로 바꿔", ("set_persona", {"name": "츤데레 친구"})),
    ("날씨 어때", None),
])
def test_offline_parse(text, expected):
    assert offline.parse(text) == expected


# ---- 대화 기억 ----------------------------------------------------------------
def test_history_resets_after_max_turns():
    agent, client, _, _ = make(*[resp(text("네.")) for _ in range(3)], cfg=AgentConfig(max_turns=2))
    for _ in range(3):
        agent.handle("안녕")
    assert [len(r["messages"]) for r in client.requests] == [1, 3, 1]


# ---- 음성 연결 ----------------------------------------------------------------
def test_runner_stop_calls_robot_and_cancels():
    gate = threading.Event()
    robot = MockRobot()
    agent, _, _, _ = make(resp(use("tidy_up")), resp(text("정리했어요.")), robot=robot)
    orig = robot.run

    def slow(tool, args):
        gate.wait(2)            # 정지어가 올 때까지 동작 중
        return orig(tool, args)

    robot.run = slow
    replies = []
    t = NS(text="다 정리해", confidence=0.9, to_event=lambda: {"type": "stt"})
    with VoiceCommandRunner(agent, on_reply=replies.append) as runner:
        runner.on_transcript(t)
        runner.on_transcript(NS(text="인사해", confidence=0.9, to_event=lambda: {"type": "stt"}))
        for _ in range(100):
            if agent._lock.locked():
                break
            threading.Event().wait(0.01)
        runner.on_stop(NS(text="멈춰", to_event=lambda: {"type": "stt"}))
        gate.set()
    assert ("stop", {}) in robot.log
    assert [r.status for r in replies] == ["cancelled"]       # 대기 중이던 "인사해"는 버려짐
    assert "gesture" not in [name for name, _ in robot.log]


# ---- eval 채점 ----------------------------------------------------------------
def test_grade():
    from app.agent.agent import AgentReply, ToolCall

    r = AgentReply("", "done", [ToolCall("find_objects", {}, {}), ToolCall("pick_and_place",
                                {"target_id": "obj_3", "place": "obj_4의 옆"}, {})])
    assert grade({"expect": "pick_and_place", "args": {"target_id": "obj_3", "place": "obj_4"}}, r)[0]
    assert not grade({"expect": "hand_over"}, r)[0]
    assert not grade({"expect": "none"}, r)[0]
    asked = AgentReply("", "asked", [ToolCall("ask_clarification", {}, {})])
    assert grade({"expect": "none"}, asked)[0]
    assert grade({"expect": "ask_clarification"}, asked)[0]
