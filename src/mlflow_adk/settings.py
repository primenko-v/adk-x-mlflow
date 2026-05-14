from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    agent_model: str = "gemini-2.0-flash-lite"
    mlflow_tracking_uri: str = "http://localhost:5000"


settings = Settings()
