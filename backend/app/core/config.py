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

    # Which classifier implementation /pipeline/classifiers.py should use for each
    # side of a request, independently -- the URL and email models have very
    # different maturity (see ml/README.md §5/§6 for why URL defaults to
    # rule_based despite the trained model's higher aggregate F1: it catastrophically
    # misclassifies ordinary bare domains like "google.com"; ml/README_EMAIL.md's
    # trained model has no equivalent known failure mode). "rule_based" needs no
    # trained model on disk; "trained" loads the corresponding joblib under
    # /ml/saved_models and falls back to rule-based heuristics if that file is
    # missing for the *other* side of a mixed request.
    url_classifier_backend: str = "rule_based"
    email_classifier_backend: str = "rule_based"

    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
