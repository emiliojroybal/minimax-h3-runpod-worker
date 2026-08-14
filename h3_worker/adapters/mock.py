from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable

from .base import GenerationAdapter
from ..media import LocalReference
from ..schemas import GenerationInput
from ..utils import aspect_dimensions


class MockAdapter(GenerationAdapter):
    def generate(self, request: GenerationInput, references: list[LocalReference], output_path: Path, progress: Callable[[int, str], None] | None = None) -> float:
        del references
        if not shutil.which("ffmpeg"):
            raise RuntimeError("Mock generation requires FFmpeg.")
        width, height = aspect_dimensions(request.target.aspect_ratio, short_edge=288)
        duration = min(3, request.target.duration_seconds)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(45, "mock_render")
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=0x171714:s={width}x{height}:r=24:d={duration}",
            "-f", "lavfi", "-i", f"sine=frequency=220:sample_rate=32000:duration={duration}",
            "-vf", "drawbox=x=iw*0.08:y=ih*0.12:w=iw*0.84:h=ih*0.76:color=0xfa6941@0.9:t=8,drawbox=x=iw*0.2:y=ih*0.3:w=iw*0.6:h=ih*0.4:color=0xd5ec72@0.22:t=fill",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "96k", "-shortest", str(output_path),
        ]
        subprocess.run(command, check=True, timeout=60)
        if progress:
            progress(88, "encoding_video")
        return float(duration)
