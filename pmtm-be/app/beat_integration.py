"""Integration helper connecting beat/beat_analysis.py with pmtm-be data structures."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure beat directory is accessible in sys.path
BEAT_DIR = Path(__file__).resolve().parent.parent.parent / "beat"
if BEAT_DIR.exists() and str(BEAT_DIR) not in sys.path:
    sys.path.insert(0, str(BEAT_DIR))

try:
    from beat_analysis import analyze_audio
except ImportError:
    analyze_audio = None


def run_advanced_beat_analysis(
    audio_path: str | Path,
    *,
    snare_threshold: float = 0.65,
    beats_per_bar: int = 4,
    save_files: bool = False,
) -> dict[str, Any] | None:
    """Run advanced beat analysis on an audio file using beat/beat_analysis.py."""
    if analyze_audio is None:
        return None

    path = Path(audio_path)
    if not path.exists():
        return None

    try:
        return analyze_audio(
            audio_path_value=str(path),
            beats_per_bar=beats_per_bar,
            snare_confidence_threshold=snare_threshold,
            save_json=save_files,
            save_preview=False,
        )
    except Exception:
        return None
