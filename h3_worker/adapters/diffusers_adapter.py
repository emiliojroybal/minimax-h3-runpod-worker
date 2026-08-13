from __future__ import annotations

from pathlib import Path
from threading import Lock

from .base import GenerationAdapter
from ..config import settings
from ..media import LocalReference
from ..schemas import GenerationInput
from ..utils import aspect_dimensions, h3_num_frames


class DiffusersH3Adapter(GenerationAdapter):
    """Lazy official Modular Diffusers adapter. Importing this module never loads or downloads H3."""

    _pipelines: dict[str, object] = {}
    _lock = Lock()

    def _pipeline(self, mode: str):
        if mode in self._pipelines:
            return self._pipelines[mode]
        with self._lock:
            if mode in self._pipelines:
                return self._pipelines[mode]
            import torch
            from diffusers import ComponentsManager, ModularPipeline

            settings.model_cache.mkdir(parents=True, exist_ok=True)
            manager = ComponentsManager()
            manager.enable_auto_cpu_offload(
                device="cuda",
                memory_reserve_margin="12GB",
            )
            pipeline = ModularPipeline.from_pretrained(
                settings.model_id,
                workflow=mode,
                components_manager=manager,
                cache_dir=str(settings.model_cache),
            )
            # This is the only call that may fetch model components, and it occurs only
            # after a production inference request reaches a GPU worker.
            pipeline.load_components(dtype=torch.bfloat16)
            self._pipelines[mode] = pipeline
            return pipeline

    def generate(self, request: GenerationInput, references: list[LocalReference], output_path: Path) -> float:
        import torch
        from PIL import Image
        from diffusers.modular_pipelines.minimax_h3 import (
            MiniMaxH3AudioReference,
            MiniMaxH3ImageReference,
            MiniMaxH3VideoReference,
        )
        from diffusers.utils.export_utils import encode_video

        pipeline = self._pipeline(request.mode)
        frames = h3_num_frames(request.target.duration_seconds, request.target.fps)
        width, height = aspect_dimensions(request.target.aspect_ratio, request.target.short_edge)
        kwargs: dict[str, object] = {
            "prompt": request.resolved_prompt,
            "num_frames": frames,
            "height": height,
            "width": width,
            "num_inference_steps": request.inference_steps,
            "generator": torch.Generator(device="cpu").manual_seed(request.seed),
            "output": ["videos", "audio", "sampling_rate"],
        }

        if request.mode == "fl2va":
            first = next((item for item in references if item.reference.role == "first_frame"), None)
            last = next((item for item in references if item.reference.role == "last_frame"), None)
            if first:
                kwargs["image"] = Image.open(first.path).convert("RGB")
            if last:
                kwargs["last_image"] = Image.open(last.path).convert("RGB")
        elif request.mode == "ref2va":
            converted: list[object] = []
            for item in references:
                if item.reference.kind == "image":
                    converted.append(MiniMaxH3ImageReference.from_file(str(item.path)))
                elif item.reference.kind == "video":
                    converted.append(MiniMaxH3VideoReference.from_file(str(item.path)))
                else:
                    converted.append(MiniMaxH3AudioReference.from_file(str(item.path)))
            kwargs["references"] = converted

        result = pipeline(**kwargs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encode_video(
            result["videos"][0],
            fps=request.target.fps,
            output_path=str(output_path),
            audio=result["audio"][0],
            audio_sample_rate=result["sampling_rate"],
        )
        return frames / request.target.fps
