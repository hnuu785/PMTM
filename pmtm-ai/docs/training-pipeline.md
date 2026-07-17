# PMTM AI Training Pipeline

이 문서는 현재 `pmtm-ai` 코드 기준 학습 파이프라인을 도식화한 것이다.

## Assumptions

- 실행 엔트리포인트는 `run_training.py`이다.
- 기본 베이스 모델은 `Qwen/Qwen2.5-3B-Instruct`이며, `PMTM_MODEL_ID`로 바꿀 수 있다.
- `PMTM_EXPERIMENT_NAME`을 주면 `models/<experiment>`와 `outputs/<experiment>` 아래에 결과가 저장된다.
- `PMTM_DATA_DIR`, `PMTM_MODELS_DIR`, `PMTM_OUTPUTS_DIR`로 데이터/모델/체크포인트 루트 경로를 바꿀 수 있다.

## End-to-End Flow

```mermaid
flowchart TD
    CLI["python run_training.py<br/>--stage all|sft|sanity|grpo|eval"]
    Paths["app.paths<br/>MODEL_ID / DATA_DIR / MODELS_DIR / OUTPUTS_DIR"]
    GPU["check_gpu()<br/>CUDA, memory, bf16 확인"]
    Test["run_phonetics_test()<br/>tests/test_phonetics.py"]
    Prep["prepare_dataset()<br/>app.training.prepare_dataset"]
    SFT["run_sft()<br/>app.training.sft_qwen.train_sft"]
    Plot1["save_loss_plots_if_possible()<br/>plot_training_loss.py"]
    Sanity["reward_sanity_check()<br/>SFT 샘플 생성 + reward 분포 확인"]
    GRPO["run_grpo()<br/>app.training.grpo_qwen.train_grpo"]
    Plot2["save_loss_plots_if_possible()<br/>plot_training_loss.py"]
    Eval["run_eval()<br/>샘플 프롬프트 생성 평가"]

    CLI --> Paths --> GPU
    GPU --> Test --> Prep --> SFT --> Plot1 --> Sanity --> GRPO --> Plot2 --> Eval

    CLI -.-> Prep
    CLI -.-> Sanity
    CLI -.-> GRPO
    CLI -.-> Eval
```

## Data And Model Flow

```mermaid
flowchart LR
    RawCSV["data/merged_final_dataset_analyzed.csv<br/>artist, lyrics, audio features, rhyme_density"]
    Prepare["prepare_dataset_v3.py<br/>clean_lines → make_chunks<br/>8마디 SFT 샘플 생성"]
    Jsonl["data/prepared_dataset_v3.jsonl<br/>{ messages }"]

    Base["MODEL_ID<br/>Qwen/Qwen2.5-3B-Instruct 기본값"]
    SFTTrain["sft_qwen.py<br/>4-bit NF4 + LoRA<br/>Trainer"]
    SFTOut["models[/experiment]/sft_rap_qwen<br/>SFT LoRA adapter + tokenizer"]
    SFTCkpt["outputs[/experiment]/sft_qwen<br/>checkpoint-*"]

    PromptBuild["grpo_qwen.py build_prompts()<br/>rhyme_density 상위 TOP_N=200<br/>8마디 messages/chat-template prompt"]
    Reward["rhyme_reward()<br/>라임 점수 + 길이(8마디)<br/>중복/연속반복 penalty"]
    GRPOTrain["GRPOTrainer<br/>SFT adapter에서 시작<br/>num_generations=4"]
    GRPOOut["models[/experiment]/grpo_rap_qwen<br/>GRPO adapter"]
    GRPOCkpt["outputs[/experiment]/grpo_qwen<br/>checkpoint-*"]

    Plots["outputs[/experiment]/plots<br/>sft_qwen_loss.png<br/>grpo_qwen_loss.png"]
    Inference["app.inference.generate<br/>기본 adapter: grpo_rap_qwen"]

    RawCSV --> Prepare --> Jsonl
    Jsonl --> SFTTrain
    Base --> SFTTrain
    SFTTrain --> SFTOut
    SFTTrain --> SFTCkpt

    RawCSV --> PromptBuild
    SFTOut --> GRPOTrain
    Base --> GRPOTrain
    PromptBuild --> GRPOTrain
    Reward --> GRPOTrain
    GRPOTrain --> GRPOOut
    GRPOTrain --> GRPOCkpt

    SFTCkpt --> Plots
    GRPOCkpt --> Plots
    GRPOOut --> Inference
```

## Stage Behavior

| Stage | Command | Main work | Required input | Main output |
|---|---|---|---|---|
| `all` | `python run_training.py` | phonetics test, dataset prep, SFT, reward sanity, GRPO, eval | GPU, CSV dataset | SFT/GRPO adapters, checkpoints, loss plots |
| `sft` | `python run_training.py --stage sft` | build messages-format SFT JSONL if missing, train SFT LoRA | `data/merged_final_dataset_analyzed.csv` | `models[/experiment]/sft_rap_qwen` |
| `sanity` | `python run_training.py --stage sanity` | generate 20 completions with SFT and print reward stats | SFT adapter | reward mean/stdev/min/max |
| `grpo` | `python run_training.py --stage grpo` | train GRPO from SFT adapter | SFT adapter, analyzed CSV | `models[/experiment]/grpo_rap_qwen` |
| `eval` | `python run_training.py --stage eval` | generate sample lyrics | GRPO adapter, or SFT fallback | console samples |

## Key Checks

- SFT is skipped when `models[/experiment]/sft_rap_qwen` already exists unless `--force` is passed.
- SFT and GRPO resume from the latest `outputs[/experiment]/*/checkpoint-*` checkpoint.
- `reward_sanity_check()` should be run before GRPO unless deliberately skipped with `--skip-sanity`.
- `plot_training_loss.py` reads `trainer_state.json` files under checkpoint directories, so plots appear only after checkpoints with loss history exist.
