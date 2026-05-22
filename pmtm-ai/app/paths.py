"""Shared filesystem paths for the PMTM AI workspace."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = PROJECT_ROOT / "app"
DATA_DIR = Path(os.getenv("PMTM_DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser()
MODELS_DIR = Path(os.getenv("PMTM_MODELS_DIR", str(PROJECT_ROOT / "models"))).expanduser()
OUTPUTS_DIR = Path(os.getenv("PMTM_OUTPUTS_DIR", str(PROJECT_ROOT / "outputs"))).expanduser()
TESTS_DIR = PROJECT_ROOT / "tests"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B"
MODEL_ID = os.getenv("PMTM_MODEL_ID", DEFAULT_MODEL_ID)

for path in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
    path.mkdir(parents=True, exist_ok=True)
