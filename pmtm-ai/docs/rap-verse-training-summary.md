# Rap Verse Training Summary

작성일: 2026-07-11

이 문서는 PMTM 가사 생성 모델의 SFT/GRPO 학습 구조를 점검하면서 정리한 변경사항, 변경 이유, 그리고 대화에서 얻은 학습 설계 인사이트를 기록한다.

## 목표

우리가 목표로 잡은 좋은 랩 벌스는 아래 조건을 만족하는 8줄 랩 벌스다.

- 라임이 자연스럽다.
- 줄 단위 호흡이 일정하다.
- 동일 단어, 구, 끝단어 반복이 적다.
- 현재 데이터 수준에서는 과한 조건보다 라임/호흡/반복 억제를 우선한다.

초기에는 구체적 이미지나 8줄 안의 작은 전개도 목표로 논의했지만, 현재 CSV에서 기계적으로 뽑은 8줄 chunk가 항상 그 조건을 만족하지는 않았다. 그래서 SFT 프롬프트는 데이터가 실제로 지지할 수 있는 수준으로 낮췄다.

## 변경사항

### 1. `prepared_dataset_v2.jsonl` 생성

새 SFT 데이터셋을 만들었다.

- 파일: `pmtm-ai/data/prepared_dataset_v2.jsonl`
- 생성기: `pmtm-ai/app/training/prepare_dataset_v2.py`
- 후보 8줄 chunk: 5,954개
- 최종 샘플: 2,868개
- 메시지 구조: `user -> assistant`
- assistant 형식: 정확히 8줄 + `[End]`

기존 `prepared_dataset.jsonl`은 거의 모든 8줄 chunk를 학습에 넣는 구조였다. v2는 좋은 벌스 기준에 맞춰 품질 필터를 적용한다.

필터 기준:

- 낮은 라임 점수 제거
- 같은 줄 반복 제거
- 같은 끝단어 과다 반복 제거
- 한국어 비율이 너무 낮은 chunk 제거
- 줄 길이 평균이 너무 짧거나 긴 chunk 제거
- 줄 길이 편차가 과한 chunk 제거
- 2-gram, 3-gram 반복이 많은 chunk 제거
- BPM 이상치 제거
- 완전 중복 chunk 제거

생성 당시 주요 탈락 사유:

- `low_rhyme`: 2,201
- `duplicate_chunk`: 236
- `ending_repeat`: 185
- `line_repeat`: 167
- `phrase_repeat`: 127

### 2. SFT 입력을 v2 데이터셋으로 변경

`pmtm-ai/app/training/sft_qwen.py`의 SFT 입력을 아래로 바꿨다.

```python
DATA_PATH = str(DATA_DIR / "prepared_dataset_v2.jsonl")
```

`run_training.py`도 SFT 데이터 준비 단계에서 `prepared_dataset_v2.jsonl`을 확인하고, 없으면 `prepare_dataset_v2.py`를 실행하도록 바꿨다.

변경 이유:

- SFT는 좋은 벌스의 분포를 먼저 배워야 한다.
- GRPO는 SFT가 어느 정도 좋은 출력 분포에 있을 때 의미가 있다.
- 품질 필터 없는 SFT는 “좋은 랩 벌스”보다 “아무 8줄 가사 조각”을 배울 위험이 크다.

### 3. 학습/GRPO/추론 프롬프트 통일

최종 프롬프트를 아래 형태로 통일했다.

```text
BPM 90. Write exactly 8 lines of Korean rap verse. Use natural rhymes and consistent line breathing. Avoid repeating the same words, phrases, or ending words.
```

이 프롬프트는 SFT 데이터, GRPO prompt, API 추론, 로컬 CLI 추론에서 동일하게 사용된다.

변경한 경로:

- `pmtm-ai/app/lyric_prompts.py`
- `pmtm-ai/app/training/prepare_dataset_v2.py`
- `pmtm-ai/app/training/grpo_qwen.py`
- `pmtm-ai/app/inference/generate_for_api.py`
- `pmtm-ai/app/inference/generate.py`
- `pmtm-ai/run_training.py`

변경 이유:

- SFT 때 본 입력 형식과 추론 입력 형식이 다르면 모델 품질이 흔들린다.
- GRPO도 SFT와 같은 prompt 분포에서 진행되어야 한다.
- 이전에는 SFT, GRPO, API 추론, 로컬 CLI 추론이 서로 다른 prompt를 사용했다.

### 4. `system` 메시지 제거

기존 SFT 샘플은 `system -> user -> assistant` 구조였다. 현재는 `user -> assistant`만 사용한다.

변경 이유:

- system 문구가 모든 행에서 동일하게 반복되어 학습 신호로서 가치가 낮았다.
- 핵심 지시는 user prompt에 이미 포함되어 있다.
- 추론에서도 system 없이 같은 user prompt를 쓰는 편이 단순하고 일관적이다.

현재 데이터 구조:

```json
[
  {
    "role": "user",
    "content": "BPM 117. Write exactly 8 lines of Korean rap verse. Use natural rhymes and consistent line breathing. Avoid repeating the same words, phrases, or ending words."
  },
  {
    "role": "assistant",
    "content": "...\n[End]"
  }
]
```

### 5. `genre`와 `mood` 제거

기존 prompt에는 아래처럼 고정 조건이 있었다.

```text
genre Korean hip-hop, mood confident
```

v2에서는 제거했다.

변경 이유:

- 모든 샘플이 같은 `genre`, `mood`를 갖고 있으면 조건이 아니라 고정 노이즈가 된다.
- 특히 `mood confident`는 실제 모든 가사의 mood를 설명하지 못한다.
- 다양한 mood 라벨을 신뢰성 있게 만들지 않는 한, SFT 조건으로 넣지 않는 편이 낫다.

CLI 인자 `--genre`, `--mood`는 백엔드 호환성 때문에 일부 남아 있지만, 실제 프롬프트 생성에는 쓰지 않는다.

### 6. 프롬프트 강도 낮춤

한때 프롬프트에 아래 조건이 들어갔다.

```text
concrete imagery
a small progression across the verse
```

최종적으로 제거했다.

변경 이유:

- 현재 8줄 chunk가 항상 구체적 이미지나 명확한 전개를 갖고 있지 않다.
- 프롬프트가 정답 데이터보다 높은 요구를 하면 SFT 신호가 흐려진다.
- “small progression”은 LLM이 음악적 chord progression이나 과한 서사 구조로 오해할 여지가 있었다.

현재 프롬프트는 데이터가 실제로 지지하는 핵심 조건인 라임, 호흡, 반복 억제에 집중한다.

## `run_training.py` 상태

`run_training.py`는 현재 변경 방향과 맞는다.

- SFT stage는 `prepared_dataset_v2.jsonl`을 사용한다.
- 파일이 없으면 `prepare_dataset_v2.py`로 생성한다.
- reward sanity check는 GRPO와 같은 user-only prompt를 쓴다.
- eval도 `build_api_messages(bpm=90)`, `build_api_messages(bpm=95)`를 사용한다.

주의할 점:

- 기존 `models/.../sft_rap_qwen`이 있으면 SFT는 자동 스킵된다.
- v2로 새 학습하려면 새 experiment를 쓰는 편이 안전하다.

예:

```bash
python run_training.py --experiment-name exp-v2 --stage sft
python run_training.py --experiment-name exp-v2 --stage sanity
python run_training.py --experiment-name exp-v2 --stage grpo
```

## 대화에서 얻은 인사이트

### 1. BPM은 강한 텍스트 조건이 아니다

텍스트만 놓고 “이 벌스는 90 BPM용이고 저 벌스는 140 BPM용”이라고 강하게 말하기 어렵다. 같은 벌스도 래퍼가 half-time, double-time, 쉼, 발화 속도로 여러 BPM 위에 얹을 수 있다.

따라서 BPM은 정답 라벨이라기보다 약한 컨텍스트다. 더 직접적인 조건은 `flow density`, 줄 호흡, 라임 밀도, 반복 억제에 가깝다.

현재 구조에서는 BPM을 완전히 버리지는 않되, 핵심 품질 목표를 BPM 적합성보다 라임/호흡/반복 억제에 둔다.

### 2. 비트 입력의 의미는 BPM보다 beat profile에 있다

비트를 넣는 프로젝트라면 BPM만 쓰는 것은 정보 손실이 크다. 비트가 의미 있으려면 아래 정보를 추출해 generation condition으로 바꿔야 한다.

- 에너지
- 무드
- 드럼 밀도
- 여백
- half-time/double-time feel
- 악기 질감
- 섹션 감각

하지만 현재 데이터는 가사와 Spotify audio feature 중심이라, 음절 단위 타이밍이나 보컬 alignment를 학습하기 어렵다. 따라서 현재 단계의 현실적인 목표는 “비트에 정확히 박자를 맞춘 가사”가 아니라 “주어진 조건에서 좋은 랩 벌스 형태를 만드는 것”이다.

### 3. SFT와 GRPO의 역할은 분리되어야 한다

SFT의 역할:

- 좋은 벌스 예시의 분포를 학습한다.
- 기본 출력 형식, 라임감, 8줄 구조, 반복이 적은 경향을 만든다.

GRPO의 역할:

- 자동 점수화 가능한 항목을 더 강화한다.
- 라임 점수, `[End]`, 정확한 줄 수, 반복 억제 같은 목표에 적합하다.

GRPO가 구체적 이미지, 창의성, 한국어 자연스러움, 작은 전개까지 해결한다고 기대하면 reward hacking이 생길 수 있다. 그런 고차 품질은 추후 preference data나 judge 기반 평가가 더 적합하다.

### 4. 프롬프트는 데이터가 실제로 만족하는 수준이어야 한다

SFT 샘플은 “프롬프트 -> 정답 출력” 쌍이다. 프롬프트가 높은 품질을 요구하는데 정답 출력이 그 요구를 만족하지 못하면 모델은 해당 요구를 무시하는 방향으로 학습될 수 있다.

그래서 현재는 아래처럼 낮고 명확한 prompt가 더 낫다.

```text
Use natural rhymes and consistent line breathing. Avoid repeating the same words, phrases, or ending words.
```

### 5. 같은 user prompt가 반복되는 것은 무조건 문제는 아니다

`prepared_dataset_v2.jsonl`은 assistant 가사 chunk 완전 중복은 제거되어 있다. 같은 BPM prompt에 여러 답변이 붙는 구조는 “같은 요청에 대한 다양한 정답”으로 볼 수 있다.

다만 prompt가 너무 단조로우면 조건 학습이 약해질 수 있다. 향후 `flow density`, `rhyme level`, `repetition level` 같은 신뢰 가능한 라벨을 만들 수 있다면 prompt를 더 세분화할 수 있다.

### 6. 현재 단계에서 가장 효과가 큰 수정은 데이터 품질이다

모델 구조를 바꾸기 전에, SFT 데이터가 좋은 벌스 후보로 정제되어야 한다. 현재 변경의 핵심도 모델 변경이 아니라 `prepared_dataset_v2` 생성과 prompt 정합성 확보였다.

## 남은 선택지

지금 단계에서 필수는 아니지만, 다음 개선 후보는 아래와 같다.

1. 곡별 chunk cap 추가
   - 긴 곡 하나가 데이터셋을 과도하게 지배하지 않도록 제한한다.

2. near-duplicate chunk 제거 강화
   - 완전 중복은 제거했지만, 유사 chunk는 일부 남아 있을 수 있다.

3. `flow_density` 라벨 추가
   - BPM보다 의미 있는 조건이 될 수 있다.

4. GRPO reward 개선
   - 같은 끝단어 반복 penalty
   - n-gram 반복 penalty
   - 줄 길이 편차 기반 breathing score
   - dataset overlap penalty

5. preference tuning 검토
   - 구체성, 자연스러운 한국어, 창의성, 전체 벌스 완성도는 SFT/GRPO reward만으로는 한계가 있다.

## 현재 학습 실행 권장

새 데이터/프롬프트로 기존 어댑터와 섞이지 않게 새 experiment를 사용한다.

```bash
cd pmtm-ai
python run_training.py --experiment-name exp-v2 --stage sft
python run_training.py --experiment-name exp-v2 --stage sanity
python run_training.py --experiment-name exp-v2 --stage grpo
python run_training.py --experiment-name exp-v2 --stage eval
```
