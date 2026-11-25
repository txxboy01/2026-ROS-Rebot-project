# 2026-ROS-Rebot-project

## 음성 입력 모듈 (`app/audio`)

마이크 소리를 받아 발화 구간을 자르고, 한국어 텍스트로 바꿔 넘겨준다. 정지어("멈춰", "정지", "스톱", "그만")는 따로 감지해서 말이 끝나기 전에도 알린다.

```
마이크 → VAD(발화 구간) → faster-whisper(large-v3-turbo) → on_transcript(명령) / on_stop(정지어)
```

### 설치

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

- 처음 실행할 때 Whisper 모델(약 1.6 GB)을 내려받는다.
- NVIDIA GPU가 있으면 GPU를 쓰고, 없거나 CUDA 라이브러리가 안 맞으면 자동으로 CPU를 쓴다.
- 실측(RTX 4060): 모델 로드 약 4초, 3초짜리 명령 인식 약 0.25초. CPU(int8)에서는 같은 명령이 10초 이상 걸리므로 시연에는 GPU가 필요하다.

### 바로 써 보기

```bash
python -m app.audio --list-devices          # 마이크 목록
python -m app.audio                         # 말이 끝나면 자동으로 인식
python -m app.audio --ptt                   # Enter 누르고 말하고 Enter
python -m app.audio --device G733           # 마이크 이름 일부로 지정
python -m app.audio --file sample.wav       # 파일 인식
python -m app.audio --json                  # 결과를 JSON 한 줄로
python -m app.audio --model small           # 가벼운 모델 (정확도 낮음)
python -m app.audio --levels                # 소리 크기와 말소리 판단 실시간 표시 (설정 조정용)
```

말하기 시작하면 `● 듣는 중...`이 뜨고, 말이 끝나면(0.5초 조용) 지워지면서 결과가 나온다. 말하는 동안 계속 `듣는 중`이 안 사라지면 `--levels`로 쉬는 동안의 소리 크기가 `끝 문턱`보다 작은지 확인한다.

### 다른 모듈에서 쓰기

```python
from app.audio import VoiceListener

def on_command(t):
    # t.text: "빨간 블록 나한테 건네줘."
    # t.confidence: 0~1. 0.5 미만이면 "다시 말씀해 주세요" 권장
    agent.handle(t.text)

def on_stop(t):
    estop.trigger()          # LLM을 거치지 않고 바로 정지

listener = VoiceListener(on_transcript=on_command, on_stop=on_stop)
listener.start()             # 별도 스레드에서 동작. 콜백도 그 스레드에서 호출됨

listener.set_muted(True)     # 명령은 무시, 정지어는 계속 감지
listener.set_muted(False)

listener.close()
```

말하기 버튼 방식은 `VoiceListener(..., push_to_talk=True)`로 만들고 버튼에 `begin_utterance()` / `end_utterance()`를 연결한다.

`Transcript.to_event()`는 이벤트 버스와 로그(`events.jsonl`)에 쓰는 형식이다.

```json
{"type": "stt", "text": "멈춰.", "confidence": 0.855, "language": "ko",
 "start_s": 8.97, "end_s": 9.81, "latency_s": 0.231, "stop_word": "멈춰", "partial": true}
```

### 설정

`app/audio/config.py`의 `VoiceConfig`에서 바꾼다.

| 항목 | 기본값 | 설명 |
|---|---|---|
| `stt.model` | `large-v3-turbo` | Whisper 모델 |
| `stt.initial_prompt` | 로봇 명령 어휘 | 인식에 도움을 주는 단서 문장. **정지어를 넣으면 안 됨** (아래 참고) |
| `stt.stop_words` | 멈춰, 정지, 스톱, 그만 … | 부분 일치로 감지 |
| `stt.partial_interval_ms` | 400 | 말하는 도중 정지어 사전 검사 간격 (0이면 끔) |
| `stt.partial_window_ms` | 2000 | 사전 검사는 최근 이 길이만 봄. 긴 문장 끝의 "멈춰"도 말이 끝나기 전에 잡음 |
| `vad.end_silence_ms` | 600 | 이만큼 조용하면 발화 끝 |
| `vad.calibration_ms` | 500 | 시작 직후 잡음 수준 측정 시간 |

### 시험

```bash
pytest -q
```

마이크와 모델 없이 도는 단위 시험이다(VAD, 정지어, 리스너 흐름).

### 알아둘 점

- Whisper는 무음이나 잡음에서 문장을 지어낸다("시청해 주셔서 감사합니다", "아멘" 등). `config.py`의 목록으로 거르고, VAD가 무음 구간을 애초에 넘기지 않는다.
- 말소리가 아닌 소리(숨, 키보드 딸깍, "음")가 들어오면 모델은 단서 문장(`initial_prompt`)을 거의 매번 그대로 읊고, 이때 신뢰도가 0.7을 넘기도 한다. 마이크 시험에서 말하지도 않은 "드라이버 좀 건네줘."가 나온 원인이다. 그래서
  - 단서는 **명령이 아닌 설명문**으로 쓴다. 읊어도 실행 가능한 명령이 되지 않게.
  - 단서와 6글자 이상 연속으로 겹치는 결과는 버린다(`is_prompt_echo`). 실제 명령은 "빨간블록", "드라이버" 정도만 겹친다.
  - 단서에 정지어, 방향어("왼쪽")를 넣지 않는다. 단어 나열 형식도 쓰지 않는다(말하지 않은 "왼쪽"이 붙었음).
  - 말하는 중 사전 검사에는 단서를 쓰지 않는다.
- 신뢰도 0.4 미만은 버린다(숨소리, 감탄사가 0.2~0.35). 0.4~0.5는 넘기지만 틀렸을 가능성이 커서, 명령 해석 쪽에서 실행하지 말고 다시 묻는 것을 권장한다.
- 합성 음성 시험 기준으로 빠르게 뭉개 말한 "빨간거 줘"는 "빨간거죠"로 나온다. LLM이 문맥으로 처리하거나 다시 묻는다.
- 노이즈 게이트가 있는 헤드셋(Logitech G733 등)은 조용할 때 소리를 0으로 잘라 보낸다. 잡음 수준을 평균으로 잡으면 0으로 끌려가서 게이트가 잠깐 열릴 때마다 말소리로 착각하고, 발화가 끝나지 않아 여러 문장이 하나로 붙는다. 그래서 잡음 수준은 최근 3초 쉬는 구간의 상위 25% 지점으로 잡고, 발화 끝은 방금 말한 목소리 크기의 20%보다 작아지는 것으로도 판단한다.
- 정지어 감지는 보조 수단이다. 실제 안전은 물리 비상정지 스위치가 책임진다.
- `faster-whisper`의 `decode_audio`는 PyAV 19와 호환되지 않아서, WAV 파일은 표준 라이브러리로 직접 읽는다.

## 명령 해석 모듈 (`app/agent`)

인식된 문장을 Claude에 넘겨 로봇 도구 호출로 바꾸고, 로봇 제어 모듈에 실행을 맡긴다. LLM은 로봇을 직접 움직이지 않고 등록된 도구 중에서 고르기만 한다. 좌표는 만들지 않고 물체 id와 칸 이름으로만 지정한다.

```
문장 → Claude(도구 선택) → RobotBackend.run(도구, 인자) → 결과 → Claude → 화면에 표시할 답
```

### 준비

Claude API 키가 필요하다. 없으면 고정 명령어 모드(아래)로만 동작한다.

```bash
pip install -r requirements.txt
set ANTHROPIC_API_KEY=sk-ant-...      # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
```

### 바로 써 보기

로봇 없이 가짜 작업대(`MockRobot`: 빨간 블록 2개, 파란 블록, 드라이버, 렌치, 초록 공, 칸 3개)로 돈다.

```bash
python -m app.agent                   # 키보드로 명령 입력. /world /reset /stop /quit
python -m app.agent --voice           # 마이크로 명령 (app/audio 사용)
python -m app.agent --voice --ptt     # Enter 누르고 말하고 Enter
python -m app.agent --offline         # LLM 없이 고정 명령어만
python -m app.agent --no-camera       # 카메라 없는 로봇 프로필 (도구 목록이 바뀜)
python -m app.agent --persona 집사
```

출력 예 (LLM의 답 문장은 실행할 때마다 조금씩 다르다):

```
> 빨간 블록 줘
   → find_objects(color=red, kind=block)
   ← 성공: obj_1 빨간 블록(왼쪽), obj_2 빨간 블록(오른쪽)
[질문] 빨간 블록이 두 개 있어요. 어느 쪽을 드릴까요?
   1) 왼쪽
   2) 오른쪽
> 왼쪽
   → hand_over(target_id=obj_1)
   ← 성공: 빨간 블록(obj_1)를 건넸고 사용자가 받았습니다
[로봇] 왼쪽 빨간 블록 드렸습니다.
```

모든 과정은 `logs/events.jsonl`에 한 줄씩 남는다(`stt`, `llm`, `tool_call`, `tool_result`, `agent_reply`, `estop`).

### 도구 (로봇 제어 모듈과 맞출 부분)

`app/agent/tools.py`에 정의되어 있다. 로봇 프로필의 능력에 따라 LLM에 보여줄 도구가 달라진다.

| 도구 | 인자 | 필요 능력 |
|---|---|---|
| `find_objects` | color?, kind? | vision.top |
| `pick_and_place` | target_id, place | gripper |
| `tidy_up` | rule? (kind/color) | gripper, vision.top |
| `hand_over` | target_id, orientation? | gripper |
| `pick_from_slot` | slot, place | gripper, 카메라 **없을 때** |
| `push` | target_id, direction | vision.top, 집게 **없을 때** |
| `gesture` | name (greet/nod/shake_head/wave) | 항상 |
| `go_home` | - | 항상 |
| `ask_clarification` | question, options | 항상 (명령 모듈이 처리) |
| `set_persona` | name | 항상 (명령 모듈이 처리) |

로봇 제어 모듈은 `app/agent/robot.py`의 `RobotBackend`만 구현하면 된다.

```python
class MyRobot:
    capabilities = {"gripper", "vision.top"}     # 자가 점검 결과

    def run(self, tool: str, args: dict) -> dict:
        # 도구를 끝까지 실행하고 결과를 돌려준다
        return {"ok": True, "message": "obj_4를 건넸습니다"}   # 실패면 ok=False와 이유

    def stop(self) -> None:
        # 정지어가 들리면 다른 스레드에서 바로 불린다. 즉시 돌아와야 한다
        ...
```

`find_objects`의 결과는 `{"ok": True, "objects": [{"id": "obj_4", "name": "드라이버", "kind": "screwdriver", "color": "yellow", "position": "왼쪽", ...}]}` 형태로 돌려준다. LLM은 이 id로 다음 도구를 부른다.

### 다른 모듈에서 쓰기

```python
from app.audio import VoiceListener
from app.agent import CommandAgent, EventLog, VoiceCommandRunner

log = EventLog("logs/events.jsonl")
agent = CommandAgent(robot, on_event=log)
runner = VoiceCommandRunner(agent, on_reply=lambda r: show_on_screen(r.text, r.options), on_event=log)
listener = VoiceListener(on_transcript=runner.on_transcript, on_stop=runner.on_stop)

runner.start()
listener.start()
...
runner.stop()      # 화면 정지 버튼도 같은 경로
```

- 명령은 별도 스레드에서 하나씩 처리한다. LLM 응답을 기다리는 동안에도 음성 입력은 계속 정지어를 듣는다.
- 정지어가 오면 LLM을 거치지 않고 바로 `robot.stop()`을 부르고, 진행 중인 명령의 남은 도구는 실행하지 않으며, 대기 중인 명령은 버린다.
- `AgentReply.status`: `done` 완료, `asked` 되물음(선택지는 `options`), `cancelled` 정지, `rejected` 다시 말해 달라고 함, `error`, `offline`.

### 설정 (`app/agent/config.py`)

| 항목 | 기본값 | 설명 |
|---|---|---|
| `model` | `claude-opus-5-5` | |
| `effort` | `low` | 생각 깊이. 명령 해석은 low로 충분하고 빠르다. 정확도가 모자라면 medium으로 올려 eval로 비교 |
| `max_tool_calls` | 5 | 발화 한 번에 도구 호출 상한. 넘으면 멈추고 나눠서 말해 달라고 함 |
| `max_turns` | 10 | 이만큼 발화가 쌓이면 대화를 새로 시작 |
| `idle_reset_s` | 600 | 이만큼 명령이 없으면 대화를 새로 시작 |
| `min_confidence` | 0.5 | STT 신뢰도가 이보다 낮으면 LLM에 보내지 않고 다시 말해 달라고 함 |
| `refusal_fallback` | True | 모델이 요청을 거절하면 서버가 다른 모델로 다시 시도 (베타 기능) |

### 고정 명령어 모드

인터넷이 끊기거나 API 키가 없으면 LLM 없이 정해진 말만 처리한다: "제자리로", "인사해", "끄덕여", "고개 저어", "손 흔들어", "2번 칸 거 줘", "1번 칸 거 3번 칸으로 가져다 줘", "집사로 바꿔". 로봇이 이미 움직인 뒤 연결이 끊기면 같은 동작을 다시 하지 않고 상태 확인을 요청한다.

### 정확도 시험 (설계서 NFR-03)

```bash
python -m app.agent.eval              # 50개 명령, 목표 90%. Claude API를 실제로 호출하므로 요금이 든다
python -m app.agent.eval --limit 5
python -m app.agent.eval --effort medium
```

명령 목록은 `app/agent/eval_cases.json`(건네기 12, 되묻기 6, 옮기기 10, 정리 5, 복귀 3, 몸짓 4, 페르소나 4, 거절 6). 결과는 `logs/eval-<시각>.json`에 저장된다.

## 상태 화면 목업 (`mockup/index.html`)

설계서 5.7 상태 화면의 목업이다. 서버 없이 브라우저로 파일을 열면 가짜 데이터로 돈다.

- 명령: 입력창에 문장을 넣거나 예시를 누르면 말한 것으로 처리한다. 말하기 버튼은 누르는 동안 "듣는 중"이 되고, 떼면 입력창 문장을 보낸다.
- 되묻기("빨간 블록 줘"), 옮기기, 정리, 페르소나 전환, 평시 반응(손 흔들기 등), 비상 정지(정지 버튼, Space 두 번, "멈춰")를 흉내 낸다.
- "상단 카메라 분리 흉내"를 누르면 능력과 도구 목록이 바뀐다(설계서 FR-15).
- 명령 해석은 화면 안의 간단한 규칙으로 흉내만 낸다. 실제 연결 시에는 FastAPI의 WebSocket `/ws/events`에서 `logs/events.jsonl`과 같은 형식의 이벤트를 받아 화면을 갱신하면 된다.
