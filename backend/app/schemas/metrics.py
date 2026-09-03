from pydantic import BaseModel


class Metrics(BaseModel):
    total_requests: int
    phishing_count: int
    legitimate_count: int
    average_latency_ms: float | None
    throughput_rps: float | None

    labeled_sample_size: int
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1_score: float | None
