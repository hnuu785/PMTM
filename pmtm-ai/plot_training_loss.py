import argparse
import json
import os
from pathlib import Path

from app.paths import EXPERIMENT_NAME, OUTPUTS_DIR, PROJECT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save SFT/GRPO loss graphs from trainer_state.json as PNG files."
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


def load_loss_points(stage_dir: Path) -> list[tuple[int, float]]:
    points: dict[int, float] = {}
    trainer_states = sorted(stage_dir.glob("checkpoint-*/trainer_state.json"))
    for path in trainer_states:
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("log_history", []):
            step = item.get("step")
            loss = item.get("loss")
            if isinstance(step, int) and isinstance(loss, (int, float)):
                points[step] = float(loss)
    return sorted(points.items())


def save_plot(points: list[tuple[int, float]], title: str, output_path: Path) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required. Run `pip install -r requirements.txt` first."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [step for step, _ in points]
    losses = [loss for _, loss in points]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, marker="o", markersize=3, linewidth=1.5, color="#1f2937")
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def generate_loss_plots(outputs_dir: Path, experiment_name: str | None = None, output_dir: Path | None = None) -> list[Path]:
    if not outputs_dir.exists():
        raise FileNotFoundError(f"outputs dir not found: {outputs_dir}")

    plot_dir = output_dir or (outputs_dir / "plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    stages = [
        ("sft_qwen", "SFT Loss"),
        ("grpo_qwen", "GRPO Loss"),
    ]

    saved = []
    for stage_name, title in stages:
        stage_dir = outputs_dir / stage_name
        if not stage_dir.exists():
            continue

        points = load_loss_points(stage_dir)
        if not points:
            continue

        output_path = plot_dir / f"{stage_name}_loss.png"
        exp_label = experiment_name or outputs_dir.name
        save_plot(points, f"{title} ({exp_label})", output_path)
        saved.append(output_path)

    if not saved:
        raise RuntimeError(f"No loss history found under {outputs_dir}")

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
