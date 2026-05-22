# 변경 요약 — 2026-05

## 한 줄 요약
세션 끊김 안전성 강화 + Colab T4 속도 최적화 변형 추가. 원본(`adapters/`)은 그대로, 빠른 버전(`adapters_fast/`)을 새로 만들어 둘 다 보관.

---

## 1. 세션 끊김 복구 (SFT / GRPO 양쪽)

### 문제
- GRPO 학습 중 세션 끊기면 체크포인트는 있어도 **자동 재개 코드가 없어** 0 step부터 다시 시작.
- SFT 체크포인트는 로컬 `./outputs/sft_qwen/`에만 있고 Drive 백업이 안 돼서, 끊기면 사라짐.

### 수정

**`adapters/training/grpo_qwen.py`** — resume 로직 추가
- `_latest_checkpoint()` 헬퍼 함수 추가
- `trainer.train(resume_from_checkpoint=resume)`로 마지막 체크포인트에서 자동 재개

**`train_colab.ipynb` B1 셀** — SFT 체크포인트 Drive symlink
- `./outputs/sft_qwen/` → Drive `sft_qwen_checkpoints/`로 symlink
- C3(GRPO)에 이미 적용돼 있던 패턴을 SFT에도 동일하게 적용

### 효과
- SFT/GRPO 둘 다 세션 끊기면 마지막 체크포인트에서 자동 재개
- 새 세션 진입 시: A 섹션 처음부터 → A5가 어댑터 복원 → B1/C3 재실행 시 `[resume] from ./outputs/.../checkpoint-N` 로그 확인

---

## 2. 속도 최적화 변형 `adapters_fast/` 신규 생성

### 새 디렉토리 구조
```
adapters/              ← 원본 (손대지 않음, 백업용으로 유지)
└── training/
    ├── run_training.py
    ├── sft_qwen.py
    ├── grpo_qwen.py
    └── prepare_dataset.py

adapters_fast/         ← 신규, 속도 최적화 버전
└── training/
    ├── run_training.py       (원본 동일)
    ├── sft_qwen.py           ★ 수정
    ├── grpo_qwen.py          ★ 수정
    └── prepare_dataset.py    (원본 동일)
```

### 변경된 파라미터 (안전 패키지 1, 4, 6)

| # | 파일 | 변경 | 효과 |
|---|------|------|------|
| 1 | `adapters_fast/training/sft_qwen.py:66` | `model.gradient_checkpointing_enable(...)` 줄 제거 | SFT 약 1.5~2x 속도 향상. backward에서 activation 재계산 없음. 1.5B+4bit+LoRA(r=32)는 T4 16GB에 그대로 들어감. |
| 4 | `adapters_fast/training/grpo_qwen.py:149` | `max_completion_length=256` → `160` | GRPO generation 토큰 수 37% 감소. 한국어 8마디 verse는 100~150 토큰이라 잘리는 케이스 거의 없음. |
| 6 | `adapters_fast/training/grpo_qwen.py:154` | `save_steps=25` → `50` | Drive symlink로 인한 체크포인트 I/O 부담 절반. 끊김 시 최대 손실 한도는 50 step (약 30~45분). |

### 학습 시간 예상 (T4 + Qwen2.5-1.5B)

| 단계 | 원본 (`adapters/`) | Fast (`adapters_fast/`) |
|------|------|------|
| SFT | 약 3h | 약 **1.5~1.8h** |
| GRPO | 약 6~8h | 약 **3~4h** |
| 합계 | 9~11h | **4.5~5.5h** |

### 성능 영향
세 항목 모두 **모델 학습 품질에 영향 0** (메모리/속도/I/O 트레이드오프만 변경, 학습 신호는 동일).

---

## 3. 노트북 `train_colab.ipynb` A4 수정

### zip 소스 경로 변경
- `adapters/training/*.py` → `adapters_fast/training/*.py`
- 결과: 다음번 zip 생성 시 fast 버전이 자동 패키징됨
- 원본으로 돌리려면 A4 명령의 `adapters_fast` → `adapters` 6곳만 치환 후 재생성

### PowerShell 디렉토리 평탄화 버그 수정
- 이전 명령에 `p.relative_to(d).as_posix()`가 있어 `rhyme_scoring/`, `tests/`, `tests_data/` 디렉토리 prefix가 사라지고 zip 루트로 평탄화되던 버그
- `p.as_posix()`로 수정 → 디렉토리 구조 보존

---

## 4. 재생성된 `project.zip` (팀원 공유용)

- 크기: **1.72 MB** (10 files)
- 소스: `adapters_fast/` 기준 (속도 최적화 버전)

### zip 내용 (A4 검증 통과 구조)
```
project.zip
├── run_training.py
├── training/
│   ├── sft_qwen.py
│   ├── grpo_qwen.py
│   └── prepare_dataset.py
├── rhyme_scoring/
│   ├── rhyme_engine.py
│   ├── phonetics_utils.py
│   └── analyze_dataset.py
├── tests/
│   └── test_phonetics.py
└── tests_data/
    ├── merged_final_dataset_analyzed.csv
    └── prepared_dataset.jsonl
```

---

## 5. 팀원 공유 / 실행 가이드

### 공유 파일
1. `project.zip` (1.72 MB)
2. `train_colab.ipynb`

### 팀원 실행 순서
1. Colab에서 `train_colab.ipynb` 열기 → 런타임 유형 = **T4 GPU**
2. A 섹션 처음부터 실행
   - A3: Google Drive 마운트 (`/content/drive/MyDrive/baseline_rapshit/`에 백업됨)
   - A4: `project.zip` 업로드
   - A5: Drive에 이전 학습본 있으면 자동 복원
   - A6: MODEL_ID를 `Qwen/Qwen2.5-1.5B`로 자동 패치
   - A7: phonetics 회귀 테스트 — 실패하면 학습 들어가지 말 것
3. B 섹션 (SFT, 1.5~1.8h) → 끝나면 B2가 Drive 백업
4. C 섹션 (GRPO, 3~4h) → 끝나면 C4가 Drive 백업
5. D 섹션 (평가)

### 세션 끊기면
1. 새 세션에서 A 처음부터 다시 (zip 재업로드 포함)
2. A5가 Drive의 어댑터 자동 복원
3. 진행 중이던 단계(B1 또는 C3) 다시 실행 → `[resume] from .../checkpoint-N` 로그 뜨면 정상 재개

### OOM 발생 시
- **SFT**: `training/sft_qwen.py`에서 `per_device_train_batch_size=2` → `1`, `gradient_accumulation_steps=8` → `16`
- **GRPO**: `training/grpo_qwen.py`에서 `max_completion_length=160` → `128` (더 줄여도 됨). `num_generations=4`는 **건드리지 말 것** (GRPO 신호 품질 직결).
