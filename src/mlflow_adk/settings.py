import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    google_cloud_project: str
    google_cloud_location: str = "global"
    google_genai_use_vertexai: bool = True

    agent_model: str = "gemini-3.1-flash-lite"
    mlflow_tracking_uri: str = "http://localhost:5000"

    def model_post_init(self, _ctx) -> None:
        # google-adk / google-genai read these from os.environ directly, so
        # propagate the Settings values out for downstream libraries.
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", self.google_cloud_project)
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", self.google_cloud_location)
        os.environ.setdefault(
            "GOOGLE_GENAI_USE_VERTEXAI", str(self.google_genai_use_vertexai).lower()
        )


settings = Settings()
