from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SECRET_KEY = "dev-secret-change-me"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # "development" (default) only ever warns about an insecure config;
    # "production" makes app.main's startup check refuse to boot instead. Set
    # ENVIRONMENT=production in whatever actually deploys this.
    environment: str = "development"

    database_url: str = "sqlite:///./phishing_detection.db"
    secret_key: str = DEFAULT_SECRET_KEY
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # which classifier implementation /pipeline/classifiers.py should hand out.
    # "rule_based" works with no trained models on disk; "trained" loads whichever
    # RF/XGBoost model ml/training/train_url_classifier.py serialized under
    # /ml/saved_models (see pipeline/classifiers.py).
    classifier_backend: str = "rule_based"

    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
