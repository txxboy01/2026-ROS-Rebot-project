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

listener.set_muted(True)     # TTS 재생 중: 명령은 무시, 정지어는 계속 감지
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
