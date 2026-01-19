"""
Configuration Management - Centralized Settings
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application Configuration"""

    # LLM Settings
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "llama3.2:1b"

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8003
    api_reload: bool = True
    api_log_level: str = "info"

    # Service Metadata
    service_name: str = "agent3-spec-generator"
    service_version: str = "1.0.0"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Singleton pattern for settings"""
    return Settings()


# Global settings instance
settings = get_settings()
