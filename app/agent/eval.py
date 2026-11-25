"""명령 해석 정확도 시험 (설계서 NFR-03: 시나리오 명령 50개 중 90% 이상 올바른 도구 선택).

    python -m app.agent.eval                 50개 전부 (Claude API 호출, 요금 발생)
    python -m app.agent.eval --limit 5       앞 5개만
    python -m app.agent.eval --effort medium 생각 깊이 바꿔서 비교

채점: 명령마다 새 대화, 새 가짜 작업대에서 시작한다. find_objects를 뺀 첫 도구 호출이
기대한 도구이고, 적어 둔 인자가 맞으면 통과.
- expect "none": 로봇을 움직이는 도구를 부르지 않으면 통과 (되묻기는 허용)
- place 인자는 띄어쓰기를 무시하고 기대값이 들어 있으면 맞은 것으로 본다
결과는 logs/eval-<시각>.json에 저장한다.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

from .agent import AgentReply, CommandAgent
from .config import AgentConfig
from .robot import MockRobot
from .tools import ACTION_TOOLS

CASES = Path(__file__).with_name("eval_cases.json")


def _norm(v) -> str:
    return str(v).replace(" ", "")


def grade(case: dict, reply: AgentReply) -> tuple[bool, str]:
    calls = [c for c in reply.tool_calls if c.name != "find_objects"]
    first = calls[0] if calls else None
    expect = case["expect"]
    if expect == "none":
        moved = [c.name for c in calls if c.name in ACTION_TOOLS or c.name == "set_persona"]
        return (not moved, f"움직임 {moved}" if moved else "")
    if first is None:
        return False, "도구 호출 없음"
    if first.name != expect:
        return False, f"{first.name} 호출"
    for key, want in case.get("args", {}).items():
        got = first.args.get(key)
        ok = _norm(want) in _norm(got) if key == "place" else got == want
        if not ok:
            return False, f"{key}={got!r} (기대 {want!r})"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m app.agent.eval")
    p.add_argument("--limit", type=int)
    p.add_argument("--model")
    p.add_argument("--effort", choices=["low", "medium", "high"])
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    cases = json.loads(CASES.read_text(encoding="utf-8"))[: args.limit]
    rows, latencies = [], []
    for i, case in enumerate(cases, 1):
        cfg = AgentConfig(persona=case.get("persona", "툴메이트"))
        if args.model:
            cfg.model = args.model
        if args.effort:
            cfg.effort = args.effort
        robot = MockRobot()
        agent = CommandAgent(robot, cfg)
        reply = agent.handle(case["text"])
        if reply.status in ("offline", "error"):
            print(f"LLM을 쓸 수 없어 중단합니다 ({reply.status}): {reply.text}")
            print("API 키(ANTHROPIC_API_KEY)나 인터넷 연결을 확인하세요.")
            return 1
        ok, why = grade(case, reply)
        latencies.append(reply.latency_s)
        tools = " → ".join(f"{c.name}({json.dumps(c.args, ensure_ascii=False)})" for c in reply.tool_calls)
        rows.append({**case, "pass": ok, "why": why, "reply": reply.text, "tools": tools,
                     "latency_s": round(reply.latency_s, 2)})
        print(f"{i:>2} {'O' if ok else 'X'} {case['text']:<24} {reply.latency_s:5.1f}s  {why}", flush=True)

    passed = sum(r["pass"] for r in rows)
    acc = passed / len(rows)
    med = statistics.median(latencies)
    print(f"\n정확도 {passed}/{len(rows)} = {acc:.0%} (목표 90%)   처리 시간 중앙값 {med:.1f}s")
    for r in rows:
        if not r["pass"]:
            print(f"  X {r['text']}: {r['why']}  | {r['tools'] or '-'} | {r['reply']}")

    out = Path("logs") / f"eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    cfg = AgentConfig()
    out.write_text(json.dumps({"model": args.model or cfg.model, "effort": args.effort or cfg.effort,
                               "accuracy": acc, "median_latency_s": med, "cases": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"결과 저장: {out}")
    return 0 if acc >= 0.9 else 2


if __name__ == "__main__":
    raise SystemExit(main())
