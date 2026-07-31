# Flow A — 한국어 DiffSinger Flow Planner

기존 `pmtm-be`와 분리해 실험하기 위한 규칙 기반 Flow Planner이다.

## 현재 입력과 출력

입력:

1. `absolute_grid`가 들어 있는 비트 분석 JSON
2. 한 줄을 한 마디로 작성한 8줄 또는 16줄 가사 TXT
3. 장르(`boom_bap` 또는 `trap`)
4. DiffSinger 보이스뱅크 ID와 기준 음높이

출력:

```text
flow-plan.json  사람이 확인하고 비교하기 위한 음절 배치 결과
score.ds        DiffSinger에 전달할 음소·길이·음높이 입력
```

## 처리 흐름

```text
비트 분석 JSON
    ↓
16분음표 슬롯을 마디별로 묶기
    ↓
가사 한 줄을 한 마디에 연결
    ↓
한국어 G2P 및 음절→음소 변환
    ↓
발화 밀도에 따라 음절을 슬롯에 분산
    ↓
붐뱁은 2·4박, 트랩은 3박 주변 음절에 강세 표시
    ↓
음절별 slot·onsetSec·durationSec·restAfterSec 계산
    ↓
flow-plan.json과 DiffSinger score.ds 생성
```

현재 구현은 규칙 기반 프로토타입이다. 실제 킥·스네어를 오디오에서 검출하지 않고 장르별 전형적인 위치를 사용한다.

## 폴더 구조

```text
flow-a/
  input/
    beat-analysis/
      80_boombap_advanced_analysis.json
      90_boombap_advanced_analysis.json
      120_trap_advanced_analysis.json
      140_trap_advanced_analysis.json
    lyrics/
      example-8bars.txt
  output/
  README.md
  requirements.txt
  src/
    flow_planner.py
  tests/
    test_flow_planner.py
```

`input`에는 Flow Planner가 읽을 파일, `output`에는 실행 결과, `src`에는 실행 코드를 둔다.

## 설치

```powershell
cd flow-a
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 실행

```powershell
python src\flow_planner.py `
  --beat-analysis "input\beat-analysis\120_trap_advanced_analysis.json" `
  --lyrics "input\lyrics\example-8bars.txt" `
  --genre trap `
  --voicebank potg `
  --output-dir "output\120-trap"
```

기본 예제 가사는 `input/lyrics/example-8bars.txt`에 있다.

## flow-plan.json에서 확인할 부분

각 음절에 다음 값이 기록된다.

```json
{
  "syllable": "나",
  "slot": 0,
  "onsetSec": 0.1974,
  "durationSec": 0.225,
  "restAfterSec": 0.025,
  "accent": false,
  "phonemes": ["n", "a"]
}
```

- `slot`: 해당 마디 안의 0~15 위치
- `onsetSec`: 음절이 시작되는 절대 시각
- `durationSec`: 목표 발화 길이
- `restAfterSec`: 다음 음절 전까지 남는 쉼
- `accent`: 장르별 스네어 위치 주변의 강조 여부
- `phonemes`: DiffSinger 보이스뱅크에 전달할 음소

## 아직 포함하지 않은 것

- 오디오에서 실제 킥·스네어 위치 검출
- 학습 기반 플로우 생성
- 라임 단어 자동 인식
- 자연스러운 억양·피치 예측
- DiffSinger 보이스뱅크 파일
- RVC 및 최종 믹싱

이 단계에서는 같은 비트와 가사를 넣었을 때 친구의 Flow Planner와 음절 배치, 발음, 쉼, 강세를 비교하는 것이 목적이다.
