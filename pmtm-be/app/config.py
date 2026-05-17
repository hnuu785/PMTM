from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "PMTM API"
    app_env: str = "development"
    cors_origins: str = "http://localhost:3100"
    database_url: str = "postgresql://pmtm:pmtm@localhost:5433/pmtm"
    redis_url: str = "redis://localhost:6380/0"

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
