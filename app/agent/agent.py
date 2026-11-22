"""명령 해석: 인식된 문장 → Claude(도구 호출) → 로봇 제어 모듈 (설계서 5.2).

LLM은 로봇을 직접 움직이지 않는다. 등록된 도구 중 하나를 고르고, 실제 실행은
`RobotBackend.run()`이 맡는다. 좌표는 LLM이 만들지 않고 물체 id와 칸 이름으로만 지정한다.

대화 흐름
- 한 발화마다 `handle(text)`를 부른다. 도구 호출 → 결과 → 다시 LLM … 을 반복하다
  LLM이 말로 끝내면 그 문장을 돌려준다(화면에 표시할 답).
- LLM이 `ask_clarification`을 부르면 거기서 멈추고 질문을 돌려준다. 다음 `handle()`의
  문장이 그 질문의 답으로 이어진다.
- `cancel()`(정지어)이 오면 남은 도구는 실행하지 않는다.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from . import offline
from .config import AgentConfig
from .persona import PERSONAS, persona_prompt
from .robot import RobotBackend
from .tools import TOOLS_BY_NAME, ToolSpec, build_tools, validate_input

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"

RULES = """너는 작업대 위의 SO-101 로봇팔을 움직이는 음성 비서다. 사용자의 말은 음성 인식 결과라서 \
맞춤법이 틀리거나 비슷한 발음으로 잘못 적혀 있을 수 있다(예: "빨간거죠" = "빨간 거 줘"). 뜻을 \
짐작할 수 있으면 그대로 처리한다.

할 일
- 사용자의 요청을 주어진 도구로 실행한다. 로봇을 움직일 수 있는 방법은 도구뿐이다.
- 물체는 find_objects가 돌려준 id로만 지정한다. 좌표나 id를 지어내지 않는다.
- 사용자 말에 맞는 후보가 여럿이고 말(색, 종류, 왼쪽·오른쪽 등)로 하나를 고를 수 없으면 \
실행하지 말고 ask_clarification으로 되묻는다. 후보가 없으면 실행하지 말고 없다고 말한다.
- 도구로 할 수 없는 요청이거나 사람이나 물건을 위험하게 하는 요청은 실행하지 않고 짧게 이유를 말한다.
- 도구 결과가 실패면 이유를 사용자에게 짧게 전한다.
- "멈춰" 같은 정지는 다른 장치가 바로 처리하므로 너는 신경 쓰지 않는다.

답하는 방식
- 답은 화면에 글자로 표시된다. 한국어 한두 문장으로 짧게 말한다. 목록이나 마크다운은 쓰지 않는다.
- 아래 페르소나의 말투를 따른다. 페르소나가 바뀌면 바뀐 말투를 따른다."""


@dataclass
class ToolCall:
    name: str
    args: dict
    result: dict
    is_error: bool = False


@dataclass
class AgentReply:
    text: str
    status: str                   # done | asked | cancelled | rejected | error | offline
    tool_calls: list[ToolCall] = field(default_factory=list)
    options: list[str] = field(default_factory=list)   # status == asked 일 때 선택지
    latency_s: float = 0.0

    def to_event(self) -> dict:
        return {"type": "agent_reply", "text": self.text, "status": self.status,
                "options": self.options, "latency_s": round(self.latency_s, 3),
                "tools": [c.name for c in self.tool_calls]}


class CommandAgent:
    def __init__(self, robot: RobotBackend, cfg: AgentConfig | None = None, client=None,
                 on_event: Callable[[dict], None] | None = None):
        """
        robot: 도구를 실제로 실행할 로봇 제어 모듈 (개발 중에는 MockRobot)
        client: anthropic.Anthropic 객체. None이면 첫 명령 때 만든다(시험에서는 가짜를 넣는다)
        on_event: 이벤트 로그/상태 화면으로 보낼 콜백
        """
        self.cfg = cfg or AgentConfig()
        if self.cfg.persona not in PERSONAS:
            raise ValueError(f"모르는 페르소나: {self.cfg.persona}. 가능: {list(PERSONAS)}")
        self.robot = robot
        self.persona = self.cfg.persona
        self.on_event = on_event or (lambda e: None)
        self._client = client
        self._cancel = threading.Event()
        self._lock = threading.Lock()          # handle()은 한 번에 하나씩
        self.messages: list[dict] = []
        self._turns = 0
        self._last_t = 0.0
        self._system = ""
        self._tools: list[ToolSpec] = []
        self._robot_moved = False

    # ---- 바깥에서 부르는 것 ---------------------------------------------------
    def handle(self, text: str, confidence: float = 1.0) -> AgentReply:
        """인식된 문장 하나를 처리하고, 화면에 보여줄 답을 돌려준다."""
        with self._lock:
            t0 = time.monotonic()
            self._cancel.clear()
            reply = self._handle(text.strip(), confidence)
            reply.latency_s = time.monotonic() - t0
            self.on_event(reply.to_event())
            return reply

    def cancel(self) -> None:
        """정지어. 다른 스레드에서 불러도 된다. 진행 중인 명령의 남은 도구를 건너뛴다."""
        self._cancel.set()

    def reset(self) -> None:
        """대화를 새로 시작한다. 페르소나는 유지."""
        self.messages = []
        self._turns = 0

    @property
    def waiting_answer(self) -> bool:
        """되물은 질문의 답을 기다리는 중인지."""
        return bool(self.messages) and self.messages[-1]["role"] == "user"

    # ---- 내부 -------------------------------------------------------------
    def _handle(self, text: str, confidence: float) -> AgentReply:
        if not text:
            return AgentReply("", "rejected")
        if confidence < self.cfg.min_confidence:
            return AgentReply("잘 못 들었어요. 다시 말씀해 주세요.", "rejected")
        if self.cfg.offline:
            return self._offline(text)

        now = time.monotonic()
        if self.messages and not self.waiting_answer and (
            self._turns >= self.cfg.max_turns or now - self._last_t > self.cfg.idle_reset_s
        ):
            self.reset()
        self._last_t = now
        if not self.messages:
            self._start_session()

        saved = len(self.messages), self._copy_last()
        self._append_user([{"type": "text", "text": text}])
        self._turns += 1
        self._robot_moved = False
        try:
            return self._loop()
        except Exception as e:  # noqa: BLE001 - 아래에서 종류별로 나눔
            if self._robot_moved:
                # 로봇은 이미 움직였다. 기록은 그대로 두고(도구 결과까지 들어가 있음) 알리기만 한다.
                log.exception("작업 중 LLM 호출 실패")
                self.on_event({"type": "agent_error", "error": repr(e)})
                return AgentReply("작업 도중 연결이 끊겼어요. 로봇 상태를 확인해 주세요.", "error")
            self._rollback(*saved)
            self._turns -= 1
            if isinstance(e, _Refused):
                return AgentReply("그 요청은 처리할 수 없어요.", "rejected")
            if _no_credentials(e):
                # 키가 없으면 다음 명령도 마찬가지니 고정 명령어 모드로 아예 전환한다
                log.warning("Claude API 키가 없어 고정 명령어 모드로 전환합니다 (ANTHROPIC_API_KEY 설정 필요)")
                self.cfg.offline = True
                self.on_event({"type": "agent_error", "error": "no_credentials", "fallback": "offline"})
                return self._offline(text)
            if _is_network_error(e):
                log.warning("LLM 연결 실패, 이번 명령은 고정 명령어로 처리: %s", e)
                self.on_event({"type": "agent_error", "error": repr(e), "fallback": "offline"})
                return self._offline(text)
            log.exception("명령 처리 실패")
            self.on_event({"type": "agent_error", "error": repr(e)})
            return AgentReply("처리 중 오류가 났어요. 다시 말씀해 주세요.", "error")

    def _start_session(self) -> None:
        self._tools = build_tools(self.robot.capabilities)
        caps = ", ".join(sorted(self.robot.capabilities)) or "없음"
        self._system = f"{RULES}\n\n로봇 능력: {caps}\n\n{persona_prompt(self.persona)}"

    def _loop(self) -> AgentReply:
        calls: list[ToolCall] = []
        n_calls = 0
        while True:
            resp = self._create()
            if resp.stop_reason == "refusal":
                raise _Refused()
            if resp.stop_reason == "max_tokens":
                raise RuntimeError("응답이 max_tokens에서 잘림")
            self.messages.append({"role": "assistant", "content": resp.content})
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses:
                return AgentReply(_text_of(resp.content), "done", calls)

            results, asked = [], None
            for use in uses:
                n_calls += 1
                if self._cancel.is_set():
                    out, err = {"ok": False, "message": "사용자가 정지시켜 실행하지 않음"}, True
                elif n_calls > self.cfg.max_tool_calls:
                    out, err = {"ok": False, "message": "도구 호출 한도 초과로 실행하지 않음"}, True
                else:
                    out, err, q = self._run_tool(use.name, use.input)
                    asked = asked or q
                calls.append(ToolCall(use.name, dict(use.input or {}), out, err))
                results.append({"type": "tool_result", "tool_use_id": use.id,
                                 "content": json.dumps(out, ensure_ascii=False), "is_error": err})
            self._append_user(results)

            if self._cancel.is_set():
                return AgentReply("멈췄습니다.", "cancelled", calls)
            if asked:
                return AgentReply(asked["question"], "asked", calls, list(asked.get("options", [])))
            if n_calls >= self.cfg.max_tool_calls:
                return AgentReply("한 번에 하기엔 복잡해요. 나눠서 말씀해 주세요.", "error", calls)

    def _create(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(timeout=self.cfg.timeout_s, max_retries=self.cfg.max_retries)
        kw = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=[{"type": "text", "text": self._system, "cache_control": {"type": "ephemeral"}}],
            tools=[t.schema() for t in self._tools],
            messages=self.messages,
            output_config={"effort": self.cfg.effort},
            cache_control={"type": "ephemeral"},
        )
        if self.cfg.refusal_fallback:
            kw.update(betas=[FALLBACK_BETA], fallbacks="default")
        t0 = time.monotonic()
        resp = self._client.beta.messages.create(**kw)
        usage = getattr(resp, "usage", None)
        self.on_event({"type": "llm", "stop_reason": resp.stop_reason,
                       "latency_s": round(time.monotonic() - t0, 3),
                       "input_tokens": getattr(usage, "input_tokens", None),
                       "cache_read": getattr(usage, "cache_read_input_tokens", None),
                       "output_tokens": getattr(usage, "output_tokens", None)})
        return resp

    def _run_tool(self, name: str, args) -> tuple[dict, bool, dict | None]:
        """(결과, 오류 여부, 되묻기 내용)"""
        spec = TOOLS_BY_NAME.get(name)
        if spec is None or spec not in self._tools:
            return {"ok": False, "message": f"'{name}' 도구는 지금 쓸 수 없음"}, True, None
        problem = validate_input(spec, args)
        if problem:
            return {"ok": False, "message": f"잘못된 인자: {problem}"}, True, None
        self.on_event({"type": "tool_call", "name": name, "args": args})

        if name == "ask_clarification":
            out = {"ok": True, "message": "질문을 화면에 표시함. 사용자의 답은 이어지는 메시지로 온다."}
            return out, False, args
        if name == "set_persona":
            self.persona = args["name"]
            out = {"ok": True, "message": f"페르소나를 바꿈. 이제부터 이 말투를 따른다. {persona_prompt(self.persona)}"}
            self.on_event({"type": "persona", "name": self.persona})
            return out, False, None

        self._robot_moved = True
        try:
            out = self.robot.run(name, args)
        except Exception as e:  # noqa: BLE001 - 로봇 쪽 예외도 LLM에 실패로 알린다
            log.exception("도구 실행 실패: %s", name)
            out = {"ok": False, "message": f"실행 중 오류: {e}"}
        if not isinstance(out, dict):
            out = {"ok": False, "message": f"로봇 모듈이 딕셔너리가 아닌 값을 돌려줌: {out!r}"}
        self.on_event({"type": "tool_result", "name": name, **out})
        return out, not out.get("ok", False), None

    def _offline(self, text: str) -> AgentReply:
        parsed = offline.parse(text)
        if parsed is None:
            return AgentReply(offline.HELP, "offline")
        name, args = parsed
        if name == "set_persona":
            self.persona = args["name"]
            self.on_event({"type": "persona", "name": self.persona})
            return AgentReply(f"{self.persona}(으)로 바꿨어요.", "offline",
                              [ToolCall(name, args, {"ok": True})])
        self.on_event({"type": "tool_call", "name": name, "args": args, "offline": True})
        try:
            out = self.robot.run(name, args)
        except Exception as e:  # noqa: BLE001
            out = {"ok": False, "message": f"실행 중 오류: {e}"}
        msg = out.get("message", "")
        text_out = msg if out.get("ok") else f"못 했어요. {msg}"
        return AgentReply(text_out, "offline", [ToolCall(name, args, out, not out.get("ok"))])

    # 대화 기록 다루기. 이미 보낸 메시지는 고치지 않고, 끝에 붙이기만 한다.
    def _append_user(self, content: list[dict]) -> None:
        if self.messages and self.messages[-1]["role"] == "user":
            # 되묻기/정지 뒤의 도구 결과 메시지에 이어 붙인다 (user 메시지가 연달아 오지 않게)
            self.messages[-1] = {"role": "user", "content": [*self.messages[-1]["content"], *content]}
        else:
            self.messages.append({"role": "user", "content": content})

    def _copy_last(self):
        return dict(self.messages[-1]) if self.messages else None

    def _rollback(self, length: int, last) -> None:
        del self.messages[length:]
        if last is not None:
            self.messages[-1] = last


class _Refused(Exception):
    pass


def _text_of(content) -> str:
    return " ".join(b.text for b in content if b.type == "text").strip()


def _no_credentials(e: Exception) -> bool:
    """API 키나 로그인 정보가 아예 없음 (SDK가 요청 전에 TypeError를 낸다)."""
    return isinstance(e, TypeError) and "authentication" in str(e)


def _is_network_error(e: Exception) -> bool:
    """LLM을 못 쓰는 상황(인터넷 끊김, 키 오류, 서버 장애)이면 True → 고정 명령어로 대체."""
    try:
        import anthropic
    except ImportError:
        return True
    if isinstance(e, (anthropic.APIConnectionError, anthropic.AuthenticationError)):
        return True   # APITimeoutError는 APIConnectionError의 하위 클래스
    return isinstance(e, anthropic.APIStatusError) and e.status_code >= 500
