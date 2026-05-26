"""Shared filesystem paths for the PMTM AI workspace."""

import os
import re
from pathlib import Path


def _sanitize_experiment_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
    cleaned = cleaned.strip(".-")
    return cleaned or "experiment"


def _resolve_results_dir(env_var: str, default_dir: Path) -> Path:
    root_dir = Path(os.getenv(env_var, str(default_dir))).expanduser()
    experiment_name = os.getenv("PMTM_EXPERIMENT_NAME", "").strip()
    if experiment_name:
        return root_dir / _sanitize_experiment_name(experiment_name)
    return root_dir


PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = PROJECT_ROOT / "app"
DATA_DIR = Path(os.getenv("PMTM_DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser()
MODELS_DIR = _resolve_results_dir("PMTM_MODELS_DIR", PROJECT_ROOT / "models")
OUTPUTS_DIR = _resolve_results_dir("PMTM_OUTPUTS_DIR", PROJECT_ROOT / "outputs")
TESTS_DIR = PROJECT_ROOT / "tests"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B"
MODEL_ID = os.getenv("PMTM_MODEL_ID", DEFAULT_MODEL_ID)
EXPERIMENT_NAME = os.getenv("PMTM_EXPERIMENT_NAME", "").strip() or None

for path in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
    path.mkdir(parents=True, exist_ok=True)
