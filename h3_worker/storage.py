from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import settings


@dataclass(frozen=True)
class StoredOutput:
    output_url: str | None
    output_path: str | None
    storage: str


def store_output(source: Path, client_job_id: str) -> StoredOutput:
    object_key = f"outputs/{client_job_id}.mp4"
    if settings.has_s3:
        import boto3
        client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            region_name=settings.s3_region,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
        )
        client.upload_file(str(source), settings.s3_bucket, object_key, ExtraArgs={"ContentType": "video/mp4"})
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": object_key},
            ExpiresIn=settings.signed_url_ttl,
        )
        return StoredOutput(url, None, "s3")

    volume = Path("/runpod-volume/outputs")
    destination_dir = volume if volume.parent.exists() else settings.output_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{client_job_id}.mp4"
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return StoredOutput(None, str(destination), "network-volume" if destination_dir == volume else "local")
