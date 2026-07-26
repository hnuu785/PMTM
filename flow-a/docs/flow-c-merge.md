# Flow-C 병합 설명서

`flow_planner_merged.py`는 기존 Flow-A와 Flow-B를 덮어쓰지 않고 기능 단위로
연결한 통합 플래너다.

## 병합 원칙

```text
Flow-A
  g2pk2 발음 정규화
  POTG 음소 변환
  음소별 duration
  CLI와 8/16줄 입력 검증

Flow-B
  Kiwi 형태소·품사·어절 분석
  발음형 라임 키
  내용어·기능어 기반 stress
  가사 분할과 stress 기반 슬롯 할당
  MIDI pitch
  genre/stress/snare micro offset

Flow-C 신규 연결
  원문–발음 alignment
  micro offset 이후 최종 onset/duration/rest 확정
  최종 타이밍을 score.ds까지 전달
```

Flow-B의 `MICRO_OFFSET_RULES`, 신뢰도 기준과 최대 ±15ms 제한은 변경하지
않았다. 기존 Flow-B에서는 micro timing이 `flow_plan.json`에만 기록되고
DiffSinger export는 원래 시간을 사용했지만, Flow-C는 `finalStartSec`와
`finalDurationSec`를 `score.ds`에 전달한다.

Kiwi는 Flow-B의 `<0.22` 제한 대신 Python 3.14 Windows wheel을 제공하는
`kiwipiepy>=0.23.2,<0.24`를 사용한다. 통합 코드가 사용하는 공개 API는
`Kiwi().tokenize()`이며 회귀 테스트와 실제 실행으로 호환성을 확인한다.

## 전체 처리 흐름

```text
beat-analysis.json
  BPM / Downbeat / absolute_grid / onset
  optional snare_detection.events
↓
8줄 또는 16줄 원문 가사
↓
Kiwi 형태소·품사·어절 분석
g2pk2 실제 발음 생성
↓
원문 음절 ↔ 발음 음절 alignment
↓
발음형 라임 + 품사/어절/문장 끝 stress
↓
가사 분할 + 16분음표 슬롯 배치 + MIDI pitch
↓
Flow-B micro offset
↓
final onset / duration / rest 재계산
↓
POTG 음소 + 음소별 duration
↓
flow-plan.json + score.ds
```

## Beat Analysis 입력 호환성

필수 필드는 다음과 같다.

```json
{
  "audio_file": "120_trap.mp3",
  "bpm": {"fixed_integer": 120},
  "time_signature": "4/4",
  "absolute_grid": []
}
```

Flow-B Beat Analysis 결과에 아래 필드가 있으면 신뢰도 0.65 이상의 스네어
이벤트를 micro offset 규칙에 사용한다.

```json
{
  "snare_detection": {
    "events": [
      {
        "grid_slot": 136,
        "confidence": 0.91,
        "original_time": 17.21,
        "snapped_time": 17.17
      }
    ]
  }
}
```

기존 Flow-A JSON처럼 `snare_detection`이 없어도 실행된다. 이때 스네어
이벤트는 0개이고 Flow-B의 stress/genre micro offset만 적용된다.

Beat Analysis 노트북 자체는 Flow-B의
`flow/beat/beat_analysis.ipynb`에 있고, 생성된 확장 JSON을
`--beat-analysis`에 전달하는 구조다.

## 실행

```powershell
cd flow-a
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\flow_planner_merged.py `
  --beat-analysis "input\beat-analysis\120_trap_advanced_analysis.json" `
  --lyrics "input\lyrics\example-8bars.txt" `
  --genre trap `
  --voicebank potg `
  --verse-start-bar 1 `
  --output-dir "output\flow-c-120-trap"
```

## 주요 출력 필드

```json
{
  "original": "앞",
  "pronounced": "아",
  "morphemeId": 2,
  "eojeolId": 1,
  "stress": 1.15,
  "gridStartSec": 1.2,
  "microOffsetMs": -6.24,
  "finalStartSec": 1.19376,
  "finalDurationSec": 0.21,
  "restAfterSec": 0.034,
  "midiNote": 62,
  "phonemes": ["a"],
  "phonemeDurations": [0.21]
}
```

## 현재 경계

- 실제 스네어 검출 품질은 Beat Analysis 결과에 의존한다.
- Flow-B 샘플 120 트랩 JSON에서는 신뢰도 기준을 통과한 스네어가 0개였다.
- POTG 음소표는 Flow-A 기준을 사용한다. 실제 보이스뱅크 사전 확보 후
  최종 대조가 필요하다.
- Micro offset 계산식은 Flow-B 원본을 보존했다. 실제 스네어 시간 차이를
  직접 사용하는 방식은 이번 병합 범위에 포함하지 않았다.
- DiffSinger 보이스뱅크가 없어도 두 JSON 결과는 만들 수 있지만 합성 보컬은
  만들 수 없다.
