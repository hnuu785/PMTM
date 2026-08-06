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
                pronouncing tqdm pandas
    pip install g2pk  # 실패해도 자모 분해 폴백 동작
"""

import argparse
import os
import statistics
import subprocess
import sys

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if "--use-unsloth" in sys.argv:
    try:
        import unsloth  # noqa: F401
    except ImportError:
        pass



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
from app.lyric_prompts import build_messages

sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def check_gpu():
    # pyrefly: ignore [missing-import]
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


def save_loss_plots_if_possible() -> None:
    try:
        from plot_training_loss import generate_loss_plots

        saved = generate_loss_plots(
            outputs_dir=OUTPUTS_DIR,
            experiment_name=EXPERIMENT_NAME,
        )
    except Exception as exc:
        print(f"[WARN] loss plot 저장 실패: {exc}")
        return

    print("[plots] saved:")
    for path in saved:
        print(f"  {path}")
    print()


def prepare_dataset():
    print("=" * 60)
    print("[B1] SFT 데이터셋 준비 (붐뱁 / 트랩 분리)")
    print("=" * 60)
    out_b = DATA_DIR / "prepared_dataset_boombap.jsonl"
    out_t = DATA_DIR / "prepared_dataset_trap.jsonl"
    if out_b.exists() and out_t.exists():
        nb = sum(1 for _ in out_b.open(encoding="utf-8"))
        nt = sum(1 for _ in out_t.open(encoding="utf-8"))
        print(f"이미 존재: {out_b} ({nb} samples), {out_t} ({nt} samples)")
    else:
        from app.training.prepare_dataset import main as prep_main

        prep_main()
    assert out_b.exists() and out_t.exists(), "SFT 데이터 생성 실패"
    print()


def run_sft(genre: str | None = None, force: bool = False, use_unsloth: bool = False):
    print("=" * 60)
    print(f"[B2] SFT 학습 (장르: {genre or '전체'})")
    print("=" * 60)

    if genre in ("boombap", "trap"):
        genres = [genre]
    elif genre == "both":
        genres = ["boombap", "trap"]
    else:
        genres = ["boombap", "trap"]  # 기본적으로 붐뱁, 트랩 분리 학습 진행

    from app.training.sft_qwen import train_sft

    for g in genres:
        save_dir = MODELS_DIR / f"sft_rap_qwen_{g}" if g else MODELS_DIR / "sft_rap_qwen"
        print(f"\n--- SFT 학습 시작 ({g or '통합'}) ---")
        if save_dir.exists() and not force:
            print(f"SFT 어댑터 이미 존재: {save_dir} (재학습하려면 --force)")
            continue

        train_sft(genre=g, use_unsloth=use_unsloth)
        assert save_dir.exists(), f"SFT 학습 후 {save_dir} 가 만들어지지 않았습니다"
    print()


def resolve_sft_adapter_path(genre: str | None = None) -> str:
    """Find available SFT adapter path according to specified or default genre."""
    candidates = []
    if genre in ("boombap", "trap"):
        candidates.append(MODELS_DIR / f"sft_rap_qwen_{genre}")
    for g in ["boombap", "trap"]:
        p = MODELS_DIR / f"sft_rap_qwen_{g}"
        if p not in candidates:
            candidates.append(p)
    candidates.append(MODELS_DIR / "sft_rap_qwen")

    for cand in candidates:
        if cand.exists():
            return str(cand)

    raise AssertionError(
        f"SFT 어댑터 없음: 아래 경로 중 존재하는 어댑터 폴더가 없습니다.\n"
        + "\n".join(f"  - {c}" for c in candidates)
    )


def reward_sanity_check(genre: str | None = None):
    """GRPO 들어가기 전 SFT 모델로 20개 prompt 생성 → reward 분포 확인."""
    print("=" * 60)
    print(f"[C2] Reward sanity check (장르: {genre or '자동 감지'})")
    print("=" * 60)

    import pandas as pd
    # pyrefly: ignore [missing-import]
    import torch
    # pyrefly: ignore [missing-import]
    from peft import PeftModel
    # pyrefly: ignore [missing-import]
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from app.training.grpo_qwen import MODEL_ID, build_prompts, rhyme_reward

    sft_adapter = resolve_sft_adapter_path(genre)
    print(f"[sanity check] SFT 어댑터 로드: {sft_adapter}")

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
    model = PeftModel.from_pretrained(base, sft_adapter)
    model.eval()

    df = pd.read_csv(DATA_DIR / "merged_final_dataset_analyzed.csv")
    prompts = build_prompts(df)[:20]
    prompt_texts = [
        tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        for prompt in prompts
    ]

    tok.padding_side = "left"
    inp = tok(prompt_texts, return_tensors="pt", padding=True, truncation=True,
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
    r_mean = statistics.mean(rewards)
    r_std  = statistics.stdev(rewards)
    r_min  = min(rewards)
    r_max  = max(rewards)

    print(f"reward mean   : {r_mean:+.4f}")
    print(f"reward stdev  : {r_std:.4f}")
    print(f"reward min/max: {r_min:+.4f} / {r_max:+.4f}")

    # ── 분포 진단 ──────────────────────────────────────────────
    warnings: list[str] = []

    if r_std < 0.05:
        warnings.append(
            f"[WARNING] reward std={r_std:.4f} < 0.05\n"
            "  → 보상 분산 부족: 모든 샘플이 비슷한 점수를 받아 GRPO 학습 신호가 약합니다.\n"
            "  조치: 보상 함수 가중치 재조정 또는 rhyme/syllable 항목 강화를 검토하세요."
        )

    if r_mean < 0:
        warnings.append(
            f"[WARNING] reward mean={r_mean:+.4f} < 0\n"
            "  → 음수 편향: format/dup 페널티가 과도하게 작동 중입니다.\n"
            "  조치: format_penalty / dup_penalty 가중치를 낮추거나\n"
            "        grpo_qwen.py의 scale_rewards 설정을 확인하세요."
        )

    if warnings:
        print()
        for w in warnings:
            print(w)
        print()
        print("[SANITY] FAIL — GRPO 진행 전 위 경고를 해결하는 것을 권장합니다.")
    else:
        print()
        print(f"[SANITY] PASS — std={r_std:.4f} >= 0.05, mean={r_mean:+.4f} >= 0 (GRPO 진행 가능)")

    print("\n--- sample prompt + completion ---")
    print(prompt_texts[0])
    print(completions[0][:600])
    print()

    del base, model
    torch.cuda.empty_cache()


def run_grpo(
    genre: str | None = None,
    trace_finite: bool = False,
    use_unsloth: bool = False,
    max_steps: int = 120,
):
    print("=" * 60)
    print(f"[C3] GRPO 학습 (장르: {genre or '전체'})")
    print("=" * 60)

    if genre in ("boombap", "trap"):
        genres = [genre]
    elif genre == "both":
        genres = ["boombap", "trap"]
    else:
        genres = ["boombap", "trap"]

    from app.training.grpo_qwen import train_grpo

    for g in genres:
        sft_dir = MODELS_DIR / (f"sft_rap_qwen_{g}" if g else "sft_rap_qwen")
        print(f"\n--- GRPO 학습 시작 ({g or '통합'}) ---")
        if not sft_dir.exists():
            print(f"[WARN] SFT 어댑터 없음 ({sft_dir}) — 베이스 모델로 GRPO 진행")
        train_grpo(
            genre=g,
            trace_finite=trace_finite,
            use_unsloth=use_unsloth,
            max_steps=max_steps,
        )
    print()


def run_grpo_smoke(steps: int, genre: str | None = None, use_unsloth: bool = False):
    print("=" * 60)
    print(f"[C3-smoke] GRPO finite smoke test ({steps} steps)")
    print("=" * 60)
    sft_adapter = resolve_sft_adapter_path(genre)
    print(f"[smoke test] SFT 어댑터 사용: {sft_adapter}")
    from app.training.grpo_qwen import run_grpo_smoke_test

    run_grpo_smoke_test(max_steps=steps, genre=genre, use_unsloth=use_unsloth)
    print()


def run_eval(genre: str | None = None):
    print("=" * 60)
    print(f"[D] 생성 평가 (장르: {genre or '전체'})")
    print("=" * 60)

    # pyrefly: ignore [missing-import]
    import torch
    # pyrefly: ignore [missing-import]
    from peft import PeftModel
    # pyrefly: ignore [missing-import]
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    adapter = None
    grpo_candidates = []
    if genre in ("boombap", "trap"):
        grpo_candidates.append(MODELS_DIR / f"grpo_rap_qwen_{genre}")
    for g in ["boombap", "trap"]:
        p = MODELS_DIR / f"grpo_rap_qwen_{g}"
        if p not in grpo_candidates:
            grpo_candidates.append(p)
    grpo_candidates.append(MODELS_DIR / "grpo_rap_qwen")

    for cand in grpo_candidates:
        if cand.exists():
            adapter = str(cand)
            print(f"[eval] GRPO 어댑터 발견 및 사용: {adapter}")
            break

    if not adapter:
        print("[WARN] GRPO 어댑터 없음 — SFT 어댑터 감지 중...")
        adapter = resolve_sft_adapter_path(genre)
        print(f"[eval] SFT 어댑터 사용: {adapter}")

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
        build_messages(bpm=90),
        build_messages(bpm=95),
    ]

    for messages in test_prompts:
        p = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
        choices=["all", "sft", "sanity", "grpo-smoke", "grpo", "eval"],
        default="all",
        help="실행 단계 선택 (기본 all = 전체)",
    )
    p.add_argument(
        "--genre",
        choices=["boombap", "trap", "both", "all"],
        default="both",
        help="SFT 학습 장르 선택 (boombap, trap, both, all)",
    )
    p.add_argument("--force", action="store_true", help="SFT 어댑터 있어도 재학습")
    p.add_argument("--skip-phonetics", action="store_true", help="phonetics 회귀 테스트 스킵")
    p.add_argument("--skip-sanity", action="store_true", help="GRPO 전 reward sanity check 스킵")
    p.add_argument("--skip-eval", action="store_true", help="최종 샘플 생성 스킵")
    p.add_argument("--grpo-steps", type=int, default=120, help="GRPO 학습 max_steps 지정 (기본값: 120)")
    p.add_argument("--smoke-steps", type=int, default=10, help="GRPO smoke test step 수 (1~50)")
    p.add_argument("--trace-finite", action="store_true", help="본 GRPO 학습 중 gradient/weight finite check 활성화")
    p.add_argument("--use-unsloth", action="store_true", help="Unsloth 가속 라이브러리 활성화")
    return p.parse_args()


def main():
    args = parse_args()
    if args.experiment_name:
        os.environ["PMTM_EXPERIMENT_NAME"] = args.experiment_name

    print_result_paths()
    check_gpu()

    if args.stage == "sft":
        prepare_dataset()
        run_sft(genre=args.genre, force=args.force, use_unsloth=args.use_unsloth)
        save_loss_plots_if_possible()
        return

    if args.stage == "sanity":
        reward_sanity_check(genre=args.genre)
        return

    if args.stage == "grpo":
        run_grpo(
            genre=args.genre,
            trace_finite=args.trace_finite,
            use_unsloth=args.use_unsloth,
            max_steps=args.grpo_steps,
        )
        save_loss_plots_if_possible()
        return

    if args.stage == "grpo-smoke":
        run_grpo_smoke(args.smoke_steps, genre=args.genre, use_unsloth=args.use_unsloth)
        return

    if args.stage == "eval":
        run_eval(genre=args.genre)
        return

    # stage == "all"
    if not args.skip_phonetics:
        run_phonetics_test()
    prepare_dataset()
    run_sft(genre=args.genre, force=args.force, use_unsloth=args.use_unsloth)
    save_loss_plots_if_possible()
    if not args.skip_sanity:
        reward_sanity_check(genre=args.genre)
    run_grpo(
        genre=args.genre,
        trace_finite=args.trace_finite,
        use_unsloth=args.use_unsloth,
        max_steps=args.grpo_steps,
    )
    save_loss_plots_if_possible()
    if not args.skip_eval:
        run_eval(genre=args.genre)

    print("=" * 60)
    print("✓ 전체 파이프라인 완료")
    print(f"  SFT  adapter : {MODELS_DIR / 'sft_rap_qwen'}")
    print(f"  GRPO adapter : {MODELS_DIR / 'grpo_rap_qwen'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
