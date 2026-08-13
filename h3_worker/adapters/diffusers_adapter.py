from __future__ import annotations

import time
from pathlib import Path
from threading import Lock

from .base import GenerationAdapter
from ..config import settings
from ..media import LocalReference
from ..model_cache import missing_components, resolve_model_snapshot
from ..schemas import GenerationInput
from ..utils import aspect_dimensions, h3_num_frames


class DiffusersH3Adapter(GenerationAdapter):
    """Lazy official Modular Diffusers adapter. Importing this module never loads or downloads H3."""

    _pipelines: dict[str, object] = {}
    _manager: object | None = None
    _lock = Lock()

    @classmethod
    def _components_manager(cls):
        if cls._manager is None:
            from diffusers import ComponentsManager

            manager = ComponentsManager()
            manager.enable_auto_cpu_offload(
                device="cuda",
                memory_reserve_margin="12GB",
            )
            cls._manager = manager
        return cls._manager

    def _pipeline(self, mode: str):
        if mode in self._pipelines:
            return self._pipelines[mode]
        with self._lock:
            if mode in self._pipelines:
                return self._pipelines[mode]
            import torch
            from diffusers import ModularPipeline

            started = time.monotonic()
            settings.hub_cache.mkdir(parents=True, exist_ok=True)
            snapshot = resolve_model_snapshot(
                settings.model_id,
                settings.hub_cache,
                settings.model_path,
            )
            if snapshot is not None:
                missing = missing_components(snapshot, mode)
                if missing:
                    raise RuntimeError(
                        f"The local H3 cache is incomplete for {mode}; missing: {', '.join(missing)}. "
                        "Run preload_h3.py for this workflow before starting the endpoint."
                    )
                model_source = str(snapshot)
                print(f"[h3] loading {mode} from local snapshot {snapshot}", flush=True)
            else:
                if settings.require_local_model:
                    raise RuntimeError(
                        "No preloaded MiniMax H3 snapshot was found. Attach the prepared RunPod network volume "
                        f"and verify HF_HUB_CACHE={settings.hub_cache}. Runtime model downloads are disabled."
                    )
                model_source = settings.model_id
                print(
                    f"[h3] local snapshot not found; downloading {mode} into {settings.hub_cache}",
                    flush=True,
                )

            pipeline = ModularPipeline.from_pretrained(
                model_source,
                workflow=mode,
                components_manager=self._components_manager(),
                collection="minimax-h3",
                cache_dir=str(settings.hub_cache),
            )
            print(f"[h3] pipeline configuration ready for {mode}; loading components", flush=True)
            pipeline.load_components(dtype=torch.bfloat16)
            self._pipelines[mode] = pipeline
            print(f"[h3] {mode} components ready in {time.monotonic() - started:.1f}s", flush=True)
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
