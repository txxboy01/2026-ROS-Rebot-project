"""명령 해석 실행기. 로봇 대신 가짜 작업대(MockRobot)로 동작을 확인한다.

    python -m app.agent                  키보드로 명령 입력
    python -m app.agent --voice          마이크로 명령 (말이 끝나면 자동 인식)
    python -m app.agent --voice --ptt    Enter 누르고 말하고 Enter
    python -m app.agent --offline        LLM 없이 고정 명령어만
    python -m app.agent --no-camera      카메라 없는 로봇 프로필로 (도구 목록이 바뀜)

키보드 모드 명령: /world 작업대 상태, /reset 대화 새로 시작, /stop 정지, /quit 끝내기
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from .agent import AgentReply, CommandAgent
from .config import AgentConfig
from .events import EventLog
from .persona import PERSONAS
from .robot import MockRobot

_SHOW = {"tool_call", "tool_result", "persona", "agent_error", "llm"}


def show_reply(r: AgentReply) -> None:
    if not r.text:
        return
    tag = {"asked": "질문", "cancelled": "정지", "offline": "오프라인", "error": "오류", "rejected": "다시"}.get(r.status, "로봇")
    print(f"[{tag}] {r.text}", flush=True)
    for i, o in enumerate(r.options, 1):
        print(f"   {i}) {o}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m app.agent", description="음성/텍스트 명령 → 로봇 도구 호출")
    p.add_argument("--voice", action="store_true", help="마이크로 명령")
    p.add_argument("--ptt", action="store_true", help="(--voice) Enter로 말하기 시작/끝")
    p.add_argument("--device", help="(--voice) 마이크 이름 일부 또는 번호")
    p.add_argument("--offline", action="store_true", help="LLM 없이 고정 명령어만")
    p.add_argument("--no-camera", action="store_true", help="카메라 없는 로봇 프로필로 실행")
    p.add_argument("--persona", default="툴메이트", choices=list(PERSONAS))
    p.add_argument("--model", help="Claude 모델 (기본 claude-opus-5-5)")
    p.add_argument("--effort", choices=["low", "medium", "high"], help="생각 깊이 (기본 low)")
    p.add_argument("--log", default="logs/events.jsonl", help="이벤트 로그 파일")
    p.add_argument("-q", "--quiet", action="store_true", help="도구 호출 과정은 숨기고 답만 표시")
    args = p.parse_args(argv)

    for stream in (sys.stdin, sys.stdout, sys.stderr):   # Windows 한글 깨짐 방지
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    def echo(e: dict) -> None:
        if args.quiet or e["type"] not in _SHOW:
            return
        if e["type"] == "tool_call":
            print(f"   → {e['name']}({', '.join(f'{k}={v}' for k, v in e['args'].items())})", flush=True)
        elif e["type"] == "tool_result":
            msg = e.get("message") or ", ".join(f"{o['id']} {o['name']}({o['position']})" for o in e.get("objects", [])) or "없음"
            print(f"   ← {'성공' if e.get('ok') else '실패'}: {msg}", flush=True)
        elif e["type"] == "llm":
            print(f"   · LLM {e['latency_s']}s", flush=True)
        elif e["type"] == "persona":
            print(f"   · 페르소나: {e['name']}", flush=True)
        # agent_error는 logging 경고로 이미 표시된다

    events = EventLog(args.log, echo=echo)
    cfg = AgentConfig(offline=args.offline, persona=args.persona)
    if args.model:
        cfg.model = args.model
    if args.effort:
        cfg.effort = args.effort
    caps = {"gripper"} if args.no_camera else {"gripper", "vision.top", "vision.front"}
    robot = MockRobot(capabilities=caps)
    agent = CommandAgent(robot, cfg, on_event=events)

    print(f"가짜 작업대 ({'카메라 없음' if args.no_camera else '카메라 있음'}):\n{robot.describe()}")
    print("칸: 1번 칸, 2번 칸, 3번 칸\n")
    if args.voice:
        return run_voice(agent, events, args)
    return run_text(agent, robot)


def run_text(agent: CommandAgent, robot: MockRobot) -> int:
    print("명령을 입력하세요. /world /reset /stop /quit")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line == "/quit":
            return 0
        if line == "/world":
            print(robot.describe())
            continue
        if line == "/reset":
            agent.reset()
            print("대화를 새로 시작합니다.")
            continue
        if line == "/stop":
            robot.stop()
            agent.cancel()
            continue
        show_reply(agent.handle(line))


def run_voice(agent: CommandAgent, events: EventLog, args) -> int:
    from app.audio import VoiceConfig, VoiceListener

    from .runner import VoiceCommandRunner

    def on_reply(r: AgentReply) -> None:
        show_reply(r)

    runner = VoiceCommandRunner(agent, on_reply=on_reply, on_event=events)

    def heard(t) -> None:
        print(f"[들음] {t.text}  (신뢰도 {t.confidence:.2f})", flush=True)
        runner.on_transcript(t)

    def stopped(t) -> None:
        print(f"[정지어] {t.text}", flush=True)
        runner.on_stop(t)

    vcfg = VoiceConfig()
    if args.device:
        vcfg.audio.device = args.device
    print("음성 인식 모델 불러오는 중...", flush=True)
    listener = VoiceListener(on_transcript=heard, on_stop=stopped, cfg=vcfg, push_to_talk=args.ptt)
    with runner:
        listener.start()
        try:
            if args.ptt:
                print("준비됨. Enter 누르고 말한 뒤 다시 Enter. 끝내려면 q 입력 후 Enter.")
                while input().strip().lower() != "q":
                    listener.begin_utterance()
                    print("  듣는 중...", flush=True)
                    input()
                    listener.end_utterance()
            else:
                print("준비됨. 말하면 명령으로 처리합니다. 끝내려면 Ctrl+C.")
                while True:
                    time.sleep(0.5)
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            listener.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
