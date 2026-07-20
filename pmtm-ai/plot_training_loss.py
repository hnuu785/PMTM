import argparse
import json
import os
from pathlib import Path

from app.paths import EXPERIMENT_NAME, OUTPUTS_DIR, PROJECT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save SFT/GRPO training graphs from trainer_state.json as PNG files."
    )
    parser.add_argument(
        "--experiment-name",
        default=EXPERIMENT_NAME,
        help="실험명. 없으면 현재 OUTPUTS_DIR 자체를 직접 사용",
    )
    parser.add_argument(
        "--outputs-root",
        default=None,
        help="outputs 루트 경로. 기본값은 PMTM_OUTPUTS_DIR 또는 프로젝트 outputs",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="PNG 저장 경로. 기본값은 <outputs>/<experiment>/plots 또는 <outputs>/plots",
    )
    return parser.parse_args()


def resolve_outputs_dir(args) -> Path:
    if args.outputs_root:
        base_root = Path(args.outputs_root).expanduser()
    else:
        env_root = os.getenv("PMTM_OUTPUTS_DIR")
        if env_root:
            base_root = Path(env_root).expanduser()
        elif args.experiment_name:
            base_root = PROJECT_ROOT / "outputs"
        else:
            base_root = OUTPUTS_DIR

    if args.experiment_name:
        return base_root / args.experiment_name
    return base_root


def _load_metric_points(stage_dir: Path, key: str) -> list[tuple[int, float]]:
    """trainer_state.json 체크포인트에서 특정 metric 키의 (step, value) 목록을 반환."""
    points: dict[int, float] = {}
    trainer_states = sorted(stage_dir.glob("checkpoint-*/trainer_state.json"))
    for path in trainer_states:
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("log_history", []):
            step = item.get("step")
            value = item.get(key)
            if isinstance(step, int) and isinstance(value, (int, float)):
                points[step] = float(value)
    return sorted(points.items())


def load_loss_points(stage_dir: Path) -> list[tuple[int, float]]:
    return _load_metric_points(stage_dir, "loss")


def _save_single_plot(
    points: list[tuple[int, float]],
    title: str,
    ylabel: str,
    output_path: Path,
    color: str = "#1f2937",
) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required. Run `pip install -r requirements.txt` first."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [step for step, _ in points]
    values = [v for _, v in points]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, values, marker="o", markersize=3, linewidth=1.5, color=color)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_plot(points: list[tuple[int, float]], title: str, output_path: Path) -> None:
    """하위 호환성 유지용 래퍼."""
    _save_single_plot(points, title, "Loss", output_path)


def _save_grpo_reward_plot(
    stage_dir: Path,
    exp_label: str,
    plot_dir: Path,
) -> list[Path]:
    """GRPO reward / reward_std / kl 곡선을 각각 PNG로 저장."""
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required. Run `pip install -r requirements.txt` first."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved: list[Path] = []

    # ── reward + reward_std (shaded) ──────────────────────────────────────────
    reward_pts = _load_metric_points(stage_dir, "reward")
    std_pts = _load_metric_points(stage_dir, "reward_std")

    if reward_pts:
        steps_r = [s for s, _ in reward_pts]
        values_r = [v for _, v in reward_pts]
        std_map = dict(std_pts)

        plt.figure(figsize=(10, 5))
        plt.plot(steps_r, values_r, marker="o", markersize=3, linewidth=1.5,
                 color="#2563eb", label="reward (mean)")

        if std_map:
            stds = [std_map.get(s, 0.0) for s in steps_r]
            upper = [r + sd for r, sd in zip(values_r, stds)]
            lower = [r - sd for r, sd in zip(values_r, stds)]
            plt.fill_between(steps_r, lower, upper, alpha=0.15, color="#2563eb",
                             label="±1 std")

        plt.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
        plt.title(f"GRPO Reward ({exp_label})")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = plot_dir / "grpo_qwen_reward.png"
        plt.savefig(out, dpi=180)
        plt.close()
        saved.append(out)

    # ── kl divergence ─────────────────────────────────────────────────────────
    kl_pts = _load_metric_points(stage_dir, "kl")
    if kl_pts:
        steps_k = [s for s, _ in kl_pts]
        values_k = [v for _, v in kl_pts]

        plt.figure(figsize=(10, 5))
        plt.plot(steps_k, values_k, marker="o", markersize=3, linewidth=1.5,
                 color="#dc2626")
        plt.title(f"GRPO KL Divergence ({exp_label})")
        plt.xlabel("Step")
        plt.ylabel("KL")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = plot_dir / "grpo_qwen_kl.png"
        plt.savefig(out, dpi=180)
        plt.close()
        saved.append(out)

    return saved


def generate_loss_plots(
    outputs_dir: Path,
    experiment_name: str | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    if not outputs_dir.exists():
        raise FileNotFoundError(f"outputs dir not found: {outputs_dir}")

    plot_dir = output_dir or (outputs_dir / "plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    exp_label = experiment_name or outputs_dir.name
    saved: list[Path] = []

    # ── SFT: loss만 ───────────────────────────────────────────────────────────
    sft_dir = outputs_dir / "sft_qwen"
    if sft_dir.exists():
        points = load_loss_points(sft_dir)
        if points:
            out = plot_dir / "sft_qwen_loss.png"
            _save_single_plot(points, f"SFT Loss ({exp_label})", "Loss", out)
            saved.append(out)

    # ── GRPO: loss + reward + kl ──────────────────────────────────────────────
    grpo_dir = outputs_dir / "grpo_qwen"
    if grpo_dir.exists():
        # loss 곡선 (참고용)
        loss_pts = load_loss_points(grpo_dir)
        if loss_pts:
            out = plot_dir / "grpo_qwen_loss.png"
            _save_single_plot(loss_pts, f"GRPO Loss ({exp_label})", "Loss", out,
                              color="#6b7280")
            saved.append(out)

        # reward / kl 곡선 (핵심 지표)
        saved.extend(_save_grpo_reward_plot(grpo_dir, exp_label, plot_dir))

    if not saved:
        raise RuntimeError(f"No training history found under {outputs_dir}")

    return saved


def main():
    args = parse_args()
    outputs_dir = resolve_outputs_dir(args)
    plot_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    saved = generate_loss_plots(
        outputs_dir=outputs_dir,
        experiment_name=args.experiment_name,
        output_dir=plot_dir,
    )

    print("Saved plots:")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
