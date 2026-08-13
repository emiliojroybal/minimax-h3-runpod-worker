from __future__ import annotations

import math
from pathlib import Path
from urllib.parse import urlparse


def h3_num_frames(duration_seconds: int, fps: int = 24) -> int:
    """Snap a requested duration to H3 VisualVAE's 17*n+5 frame lattice."""
    desired = duration_seconds * fps
    steps = max(0, math.ceil((desired - 5) / 17))
    return 17 * steps + 5


def suffix_from_url(url: str, fallback: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix and len(suffix) <= 10 else fallback


def aspect_dimensions(aspect_ratio: str, short_edge: int = 768) -> tuple[int, int]:
    ratios = {
        "21:9": (21, 9), "16:9": (16, 9), "4:3": (4, 3), "1:1": (1, 1),
        "3:4": (3, 4), "9:16": (9, 16), "auto": (16, 9),
    }
    rw, rh = ratios[aspect_ratio]
    if rw >= rh:
        height = short_edge
        width = round(short_edge * rw / rh / 32) * 32
    else:
        width = short_edge
        height = round(short_edge * rh / rw / 32) * 32
    return width, height


def validate_total_durations(durations: list[tuple[str, float | None]]) -> None:
    """Enforce H3's 15-second aggregate budget per timed reference type."""
    for kind in ("video", "audio"):
        total = sum(duration or 0 for reference_kind, duration in durations if reference_kind == kind)
        if total > 15:
            raise ValueError(f"Total {kind} reference duration must not exceed 15 seconds.")
