from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./phishing_detection.db"
    secret_key: str = "dev-secret-change-me"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # which classifier implementation /pipeline/classifiers.py should hand out.
    # "rule_based" works with no trained models on disk; "rf" / "xgboost" / "bert"
    # expect serialized models under /ml/saved_models (see pipeline/classifiers.py).
    classifier_backend: str = "rule_based"

    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
