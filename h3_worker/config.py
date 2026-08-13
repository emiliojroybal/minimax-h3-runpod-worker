from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _truthy(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    inference_mode: str = os.getenv("H3_INFERENCE_MODE", "mock").strip().lower()
    model_id: str = os.getenv("H3_MODEL_ID", "MiniMaxAI/MiniMax-H3")
    model_cache: Path = Path(os.getenv("HF_HOME", "/runpod-volume/huggingface"))
    output_dir: Path = Path(os.getenv("H3_OUTPUT_DIR", "/tmp/h3-outputs"))
    allow_http_references: bool = _truthy("ALLOW_HTTP_REFERENCES")
    max_download_bytes: int = int(os.getenv("MAX_REFERENCE_BYTES", str(300 * 1024 * 1024)))
    s3_endpoint: str | None = os.getenv("OBJECT_STORAGE_ENDPOINT")
    s3_bucket: str | None = os.getenv("OBJECT_STORAGE_BUCKET")
    s3_region: str = os.getenv("OBJECT_STORAGE_REGION", "auto")
    s3_access_key: str | None = os.getenv("OBJECT_STORAGE_ACCESS_KEY_ID")
    s3_secret_key: str | None = os.getenv("OBJECT_STORAGE_SECRET_ACCESS_KEY")
    signed_url_ttl: int = int(os.getenv("OUTPUT_URL_TTL", str(7 * 24 * 60 * 60)))

    @property
    def has_s3(self) -> bool:
        return all((self.s3_endpoint, self.s3_bucket, self.s3_access_key, self.s3_secret_key))


settings = Settings()
