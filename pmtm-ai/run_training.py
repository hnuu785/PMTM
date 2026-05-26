"""한국어 랩 생성 학습 (Qwen 계열 + GRPO) — 단일 파일 실행 버전.

train_colab.ipynb의 모든 셀을 합쳐 단일 스크립트로 변환.
A6000(48GB) 같은 단일 GPU 서버에서 그대로 실행 가능.

사용법:
    # 전체 파이프라인 (의존성 체크 → SFT → reward sanity → GRPO → 샘플 생성)
    python run_training.py

    # 단계별 실행
    python run_training.py --stage sft
    python run_training.py --stage grpo
    python run_training.py --stage eval
    python run_training.py --stage sanity   # GRPO 전 reward 분포만 확인

    # 일부 단계 건너뛰기
    python run_training.py --skip-sanity --skip-eval

    # 이미 학습된 SFT 있으면 자동 스킵. 다시 돌리려면:
    python run_training.py --stage sft --force

필수 패키지 (친구 서버에서 미리 설치):
    pip install "transformers>=4.45" "accelerate>=0.34" "peft>=0.13" \
                "trl>=0.12" "bitsandbytes>=0.43" "datasets>=2.20" \
                jamo pronouncing tqdm pandas
    pip install g2pk  # 실패해도 자모 분해 폴백 동작
"""

import argparse
import os
import statistics
import subprocess
import sys

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _bootstrap_experiment_name(argv: list[str]) -> None:
    for i, arg in enumerate(argv):
        if arg.startswith("--experiment-name="):
            os.environ["PMTM_EXPERIMENT_NAME"] = arg.split("=", 1)[1]
            return
        if arg == "--experiment-name" and i + 1 < len(argv):
            os.environ["PMTM_EXPERIMENT_NAME"] = argv[i + 1]
            return


_bootstrap_experiment_name(sys.argv[1:])

from app.paths import DATA_DIR, EXPERIMENT_NAME, MODEL_ID, MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def check_gpu():
    import torch

    print("=" * 60)
    print("[A1] GPU 확인")
    print("=" * 60)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 사용 불가 — GPU 환경에서 실행하세요")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    bf16 = torch.cuda.is_bf16_supported()
    print(f"device       : {name}")
    print(f"total memory : {total:.1f} GB")
    print(f"bf16 support : {bf16}")
    print()


def run_phonetics_test():
    print("=" * 60)
    print("[A6] Phonetics 회귀 테스트")
    print("=" * 60)
    rc = subprocess.call([sys.executable, "tests/test_phonetics.py"])
    if rc != 0:
        raise RuntimeError("phonetics test 실패")
    print()


def print_result_paths():
    label = EXPERIMENT_NAME or "default"
    print("=" * 60)
    print(f"[A0] Result paths ({label})")
    print("=" * 60)
    print(f"models dir    : {MODELS_DIR}")
    print(f"outputs dir   : {OUTPUTS_DIR}")
    print()


def prepare_dataset():
    print("=" * 60)
    print("[B1] SFT 데이터셋 준비")
    print("=" * 60)
    out = DATA_DIR / "prepared_dataset.jsonl"
    if out.exists():
        n = sum(1 for _ in out.open(encoding="utf-8"))
        print(f"이미 존재: {out} ({n} samples)")
    else:
        from app.training.prepare_dataset import main as prep_main

        prep_main()
    assert out.exists(), "SFT 데이터 생성 실패"
    print()


def run_sft(force: bool = False):
    print("=" * 60)
    print("[B2] SFT 학습")
    print("=" * 60)
    save_dir = MODELS_DIR / "sft_rap_qwen"
    if save_dir.exists() and not force:
        print(f"SFT 어댑터 이미 존재: {save_dir} (재학습하려면 --force)")
        print()
        return

    from app.training.sft_qwen import train_sft

    train_sft()
    assert save_dir.exists(), f"SFT 학습 후 {save_dir} 가 만들어지지 않았습니다"
    print()


def reward_sanity_check():
    """GRPO 들어가기 전 SFT 모델로 20개 prompt 생성 → reward 분포 확인.

    - std < 0.05  → reward shaping이 약함, GRPO 학습 신호 부족 우려
    - mean < 0    → format/dup 페널티 과도, 가중치 조정 권장
    """
    print("=" * 60)
    print("[C2] Reward sanity check")
    print("=" * 60)

    import pandas as pd
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from app.training.grpo_qwen import MODEL_ID, SFT_PATH, build_prompts, rhyme_reward

    assert os.path.exists(SFT_PATH), f"SFT 어댑터 없음: {SFT_PATH}"

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, SFT_PATH)
    model.eval()

    df = pd.read_csv(DATA_DIR / "merged_final_dataset_analyzed.csv")
    prompts = build_prompts(df)[:20]

    tok.padding_side = "left"
    inp = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=384).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=256,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            pad_token_id=tok.eos_token_id,
        )
    prompt_lens = inp["input_ids"].shape[1]
    completions = [tok.decode(o[prompt_lens:], skip_special_tokens=True) for o in out]

    rewards = rhyme_reward(completions, prompts=prompts)
    print(f"reward mean   : {statistics.mean(rewards):+.4f}")
    print(f"reward stdev  : {statistics.stdev(rewards):.4f}")
    print(f"reward min/max: {min(rewards):+.4f} / {max(rewards):+.4f}")
    print("\n--- sample prompt + completion ---")
    print(prompts[0])
    print(completions[0][:600])
    print()

    del base, model
    torch.cuda.empty_cache()


def run_grpo():
    print("=" * 60)
    print("[C3] GRPO 학습")
    print("=" * 60)
    assert (MODELS_DIR / "sft_rap_qwen").exists(), (
        "SFT 어댑터 없음 — 먼저 --stage sft 로 SFT를 실행하세요"
    )
    from app.training.grpo_qwen import train_grpo

    train_grpo()
    print()


def run_eval():
    print("=" * 60)
    print("[D] 생성 평가")
    print("=" * 60)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    GRPO_PATH = str(MODELS_DIR / "grpo_rap_qwen")

    if not (MODELS_DIR / "grpo_rap_qwen").exists():
        print(f"[WARN] GRPO 어댑터 없음 ({GRPO_PATH}) — SFT 어댑터로 평가")
        adapter = str(MODELS_DIR / "sft_rap_qwen")
        assert (MODELS_DIR / "sft_rap_qwen").exists(), "SFT 어댑터도 없음"
    else:
        adapter = GRPO_PATH

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()

    test_prompts = [
        "아티스트: Tablo\nBPM: 90 | 에너지: 0.65 | 댄서빌리티: 0.70 | 라우드니스: -6.0dB | 밸런스: 0.50\n[Verse 8마디]\n",
        "아티스트: Verbal Jint\nBPM: 95 | 에너지: 0.55 | 댄서빌리티: 0.60 | 라우드니스: -7.0dB | 밸런스: 0.40\n[Verse 16마디]\n",
    ]

    for p in test_prompts:
        print("=" * 60)
        print(p)
        print("-" * 60)
        inp = tok(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tok.eos_token_id,
            )
        print(tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True))
        print()


def parse_args():
    p = argparse.ArgumentParser(description="Qwen 랩 생성 학습 (SFT + GRPO)")
    p.add_argument(
        "--experiment-name",
        default=None,
        help="결과물을 models/<name>, outputs/<name> 아래에 저장",
    )
    p.add_argument(
        "--stage",
        choices=["all", "sft", "sanity", "grpo", "eval"],
        default="all",
        help="실행 단계 선택 (기본 all = 전체)",
    )
    p.add_argument("--force", action="store_true", help="SFT 어댑터 있어도 재학습")
    p.add_argument("--skip-phonetics", action="store_true", help="phonetics 회귀 테스트 스킵")
    p.add_argument("--skip-sanity", action="store_true", help="GRPO 전 reward sanity check 스킵")
    p.add_argument("--skip-eval", action="store_true", help="최종 샘플 생성 스킵")
    return p.parse_args()


def main():
    args = parse_args()
    if args.experiment_name:
        os.environ["PMTM_EXPERIMENT_NAME"] = args.experiment_name

    print_result_paths()
    check_gpu()

    if args.stage == "sft":
        prepare_dataset()
        run_sft(force=args.force)
        return

    if args.stage == "sanity":
        reward_sanity_check()
        return

    if args.stage == "grpo":
        run_grpo()
        return

    if args.stage == "eval":
        run_eval()
        return

    # stage == "all"
    if not args.skip_phonetics:
        run_phonetics_test()
    prepare_dataset()
    run_sft(force=args.force)
    if not args.skip_sanity:
        reward_sanity_check()
    run_grpo()
    if not args.skip_eval:
        run_eval()

    print("=" * 60)
    print("✓ 전체 파이프라인 완료")
    print(f"  SFT  adapter : {MODELS_DIR / 'sft_rap_qwen'}")
    print(f"  GRPO adapter : {MODELS_DIR / 'grpo_rap_qwen'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
