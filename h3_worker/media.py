from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
from PIL import Image

from .config import settings
from .schemas import ReferenceInput
from .utils import suffix_from_url, validate_total_durations


@dataclass(frozen=True)
class LocalReference:
    reference: ReferenceInput
    path: Path
    duration_seconds: float | None = None


def _validate_remote_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ({"https", "http"} if settings.allow_http_references else {"https"}):
        raise ValueError("Reference URLs must use HTTPS.")
    if not parsed.hostname:
        raise ValueError("Reference URL has no hostname.")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError("Reference hostname could not be resolved.") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("Reference URLs may not resolve to private network addresses.")


def download_reference(reference: ReferenceInput, directory: Path) -> LocalReference:
    url = str(reference.url)
    _validate_remote_url(url)
    fallback = {"image": ".png", "video": ".mp4", "audio": ".wav"}[reference.kind]
    destination = directory / f"{reference.id}{suffix_from_url(url, fallback)}"
    total = 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(60, connect=15)) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes(1024 * 1024):
                total += len(chunk)
                if total > settings.max_download_bytes:
                    raise ValueError(f"Reference {reference.name} exceeds the download limit.")
                handle.write(chunk)
    if total == 0:
        raise ValueError(f"Reference {reference.name} is empty.")
    duration = inspect_reference(reference, destination)
    return LocalReference(reference, destination, duration)


def inspect_reference(reference: ReferenceInput, path: Path) -> float | None:
    if reference.kind == "image":
        with Image.open(path) as image:
            image.verify()
            if image.width > 8192 or image.height > 8192:
                raise ValueError(f"Image {reference.name} exceeds 8192 pixels on one side.")
        return None

    try:
        import av
        with av.open(str(path)) as container:
            duration = float(container.duration or 0) / 1_000_000
    except Exception as exc:
        raise ValueError(f"Unable to inspect {reference.kind} reference {reference.name}.") from exc
    if duration and not 2 <= duration <= 15:
        raise ValueError(f"{reference.name} must be between 2 and 15 seconds long.")
    return duration


def materialize_references(references: list[ReferenceInput], directory: Path) -> list[LocalReference]:
    local_references = [download_reference(reference, directory) for reference in references]
    validate_total_durations([
        (reference.reference.kind, reference.duration_seconds)
        for reference in local_references
    ])
    return local_references
