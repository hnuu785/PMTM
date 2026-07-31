from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "PMTM API"
    app_env: str = "development"
    cors_origins: str = "http://localhost:3100"
    database_url: str = "postgresql://pmtm:pmtm@localhost:5433/pmtm"
    redis_url: str = "redis://localhost:6380/0"
    qwen_python_path: str = "../pmtm-ai/.venv/bin/python"
    qwen_timeout_seconds: int = 120
    openai_api_key: str = ""
    openai_model: str = "gpt-5-mini"
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_timeout_seconds: int = 60
    diffsinger_python_path: str = "../pmtm-svs/.venv/bin/python"
    diffsinger_voicebank_root: str = "../pmtm-svs/voicebanks"
    diffsinger_device: str = "auto"
    diffsinger_timeout_seconds: int = 900
    rvc_python_path: str = "../Applio/.venv/bin/python"
    rvc_model_root: str = "../Applio/logs"
    rvc_device: str = "auto"
    rvc_use_index: bool = False
    rvc_timeout_seconds: int = 300


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
