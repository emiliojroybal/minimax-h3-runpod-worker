from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..media import LocalReference
from ..schemas import GenerationInput


class GenerationAdapter(ABC):
    @abstractmethod
    def generate(self, request: GenerationInput, references: list[LocalReference], output_path: Path) -> float:
        """Generate an MP4 and return its actual duration in seconds."""
