from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    agent_model: str = "gemini-3.1-flash-lite"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "adk-demo"
    mlflow_simulation_experiment: str = "adk-simulation"


settings = Settings()
