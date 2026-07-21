import json
from typing import Any
from app.schemas import DemoStatus

def update_redis_status(
    redis_client,
    key_prefix: str,
    job_id: str,
    status: DemoStatus,
    progress: float,
    timeout_seconds: int,
    **kwargs: Any,
) -> None:
    payload = {
        "jobId": job_id,
        "status": status,
        "progress": str(max(0.0, min(1.0, progress))),
    }

    # Map snake_case argument keys to camelCase Redis field names
    # and format values appropriately.
    field_mappings = {
        "bpm": ("bpm", lambda v: str(v)),
        "lyrics": ("lyrics", lambda v: v),
        "voicebank": ("voicebank", lambda v: v),
        "notes": ("notes", lambda v: json.dumps(v, ensure_ascii=False)),
        "error": ("error", lambda v: v),
        "audio_url": ("audioUrl", lambda v: v),
        "vocal_url": ("vocalUrl", lambda v: v),
        "flow_plan_url": ("flowPlanUrl", lambda v: v),
    }

    for arg_name, value in kwargs.items():
        if value is not None and arg_name in field_mappings:
            redis_field, formatter = field_mappings[arg_name]
            payload[redis_field] = formatter(value)

    redis_client.hset(f"{key_prefix}{job_id}", mapping=payload)
    redis_client.expire(f"{key_prefix}{job_id}", timeout_seconds)
