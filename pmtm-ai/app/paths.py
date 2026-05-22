"""Shared filesystem paths for the PMTM AI workspace."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TESTS_DIR = PROJECT_ROOT / "tests"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B"
MODEL_ID = os.getenv("PMTM_MODEL_ID", DEFAULT_MODEL_ID)
