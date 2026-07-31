import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings


@dataclass(frozen=True)
class RvcModelProfile:
    id: str
    label: str
    model_file: str
    index_file: str | None = None


def resolve_configured_path(value: str, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base or Path(__file__).resolve().parents[1]) / path


def list_rvc_models() -> list[dict[str, object]]:
    settings = get_settings()
    project_root = Path(__file__).resolve().parents[2]
    rvc_root = resolve_configured_path(settings.rvc_model_root, project_root / "pmtm-be")

    if not rvc_root.is_dir():
        return []

    models = []
    for item in sorted(rvc_root.iterdir()):
        if item.is_dir():
            pth_files = list(item.glob("*.pth"))
            if pth_files:
                index_files = list(item.glob("*.index"))
                models.append(
                    {
                        "id": item.name,
                        "label": item.name.upper(),
                        "model_file": pth_files[0].name,
                        "has_index": len(index_files) > 0,
                        "available": True,
                    }
                )
    return models


def render_rvc(
    input_vocal_path: Path,
    output_vocal_path: Path,
    rvc_model_id: str,
    pitch_shift: int = 0,
    f0_method: str = "rmvpe",
    use_index: bool | None = None,
) -> None:
    settings = get_settings()
    project_root = Path(__file__).resolve().parents[2]
    python_path = resolve_configured_path(settings.rvc_python_path, project_root / "pmtm-be")
    rvc_root = resolve_configured_path(settings.rvc_model_root, project_root / "pmtm-be")
    model_dir = rvc_root / rvc_model_id
    applio_dir = project_root / "Applio"
    applio_core = applio_dir / "core.py"

    if not python_path.is_file():
        raise RuntimeError(f"RVC Python runtime not found: {python_path}")
    if not applio_core.is_file():
        raise RuntimeError(f"Applio core script not found: {applio_core}")
    if not model_dir.is_dir():
        raise RuntimeError(f"RVC model directory not found: {model_dir}")

    pth_files = list(model_dir.glob("*.pth"))
    if not pth_files:
        raise RuntimeError(f"No .pth model file found in RVC model dir: {model_dir}")
    model_pth = pth_files[0]

    should_use_index = settings.rvc_use_index if use_index is None else use_index
    index_files = list(model_dir.glob("*.index"))
    index_file = index_files[0] if (index_files and should_use_index) else None

    valid_f0_methods = {"crepe", "crepe-tiny", "rmvpe", "fcpe"}
    selected_f0 = f0_method if f0_method in valid_f0_methods else "rmvpe"

    command = [
        str(python_path),
        str(applio_core),
        "infer",
        "--input-path",
        str(input_vocal_path),
        "--output-path",
        str(output_vocal_path),
        "--pth-path",
        str(model_pth),
        "--index-path",
        str(index_file) if index_file else "",
        "--pitch",
        str(pitch_shift),
        "--f0-method",
        selected_f0,
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(applio_dir),
            timeout=settings.rvc_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("RVC voice conversion timed out.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or "RVC voice conversion failed."
        raise RuntimeError(detail[-4000:]) from exc

    if not output_vocal_path.is_file() or output_vocal_path.stat().st_size == 0:
        raise RuntimeError("RVC converter did not generate output vocal WAV.")
