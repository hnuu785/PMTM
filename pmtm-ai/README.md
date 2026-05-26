# PMTM AI

`pmtm-ai`는 PMTM의 가사 생성 AI 작업을 위한 독립 워크스페이스입니다.

## Structure

- `run_training.py`: 전체 학습 파이프라인 실행 엔트리포인트
- `app/training`: SFT, GRPO, 데이터 준비 등 학습 스크립트
- `app/rhyme_scoring`: 라임 점수 계산 로직
- `tests`: phonetics/rhyme 회귀 테스트
- `data`: 학습/평가용 입력 데이터
- `models`: 베이스 모델, 어댑터, 내보낸 모델 자산
- `outputs`: 학습 중간 산출물과 체크포인트
- `train_colab.ipynb`: Colab 실행 노트북
- `CHANGES.md`: 최근 변경 기록

## Suggested Flow

1. 학습 코드는 `app/training` 아래에서 관리합니다.
2. 라임/발음 관련 유틸은 `app/rhyme_scoring`에 둡니다.
3. `pmtm-be`는 직접 학습 코드를 섞지 말고, 추론 경계만 호출합니다.

## Bootstrap

```bash
cd pmtm-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`g2pk`는 선택 의존성입니다. 로컬 macOS 환경에서는 `python-mecab-ko` 빌드가 `mecab-config` 때문에 실패할 수 있는데, 이 경우 설치하지 않아도 자모 분해 폴백으로 동작합니다.

```bash
pip install g2pk || echo "g2pk 설치 실패 - fallback 사용"
```

기본 베이스 모델은 `Qwen/Qwen2.5-1.5B`입니다. 다른 모델을 쓰려면 `PMTM_MODEL_ID` 환경변수로 덮어쓸 수 있습니다.

```bash
PMTM_MODEL_ID=Qwen/Qwen2.5-3B python run_training.py --stage sft
```

결과물만 실험별로 쌓고 싶다면 `PMTM_EXPERIMENT_NAME` 또는 `--experiment-name`을 주면 됩니다. 이 경우 `data/`는 공유하고 `models/`, `outputs/`만 실험명 하위 디렉터리로 분리됩니다.

```bash
python run_training.py --experiment-name exp-001 --stage sft
python run_training.py --experiment-name exp-001 --stage grpo
python -m app.inference.generate --artist "Tablo" --bpm 90 --energy 0.65 --danceability 0.70 --loudness -6.0 --valence 0.50 --bars 8
```

위처럼 실행하면 기본 경로 기준 결과물은 아래에 쌓입니다.

- `models/exp-001/sft_rap_qwen`
- `models/exp-001/grpo_rap_qwen`
- `outputs/exp-001/sft_qwen`
- `outputs/exp-001/grpo_qwen`

환경변수 방식도 동일합니다.

```bash
export PMTM_EXPERIMENT_NAME=exp-002
python run_training.py --stage sft
python run_training.py --stage grpo
```

학습이 끝난 뒤 체크포인트의 `trainer_state.json`에서 loss 그래프 PNG를 만들 수 있습니다.

```bash
python3 plot_training_loss.py --experiment-name exp-002
```

기본 저장 경로는 `outputs/exp-002/plots/`이며, 아래 파일을 생성합니다.

- `outputs/exp-002/plots/sft_qwen_loss.png`
- `outputs/exp-002/plots/grpo_qwen_loss.png`

## Colab + Google Drive

Colab에서 `git clone` 후 실행할 때 런타임이 끊겨도 결과를 남기려면 `models`와 `outputs`를 Drive로 보내는 게 맞습니다.

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
cd /content
git clone <YOUR_REPO_URL>
cd PMTM2/pmtm-ai
pip install -r requirements.txt
```

```bash
export PMTM_MODELS_DIR=/content/drive/MyDrive/pmtm-ai/models
export PMTM_OUTPUTS_DIR=/content/drive/MyDrive/pmtm-ai/outputs
python run_training.py --stage sft
python run_training.py --stage sanity
python run_training.py --stage grpo
python run_training.py --stage eval
```

이렇게 하면 저장 위치는 아래처럼 바뀝니다.

- SFT 최종 어댑터: `/content/drive/MyDrive/pmtm-ai/models/sft_rap_qwen`
- GRPO 최종 어댑터: `/content/drive/MyDrive/pmtm-ai/models/grpo_rap_qwen`
- SFT 체크포인트: `/content/drive/MyDrive/pmtm-ai/outputs/sft_qwen`
- GRPO 체크포인트: `/content/drive/MyDrive/pmtm-ai/outputs/grpo_qwen`
