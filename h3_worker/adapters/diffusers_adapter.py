from __future__ import annotations

import time
from pathlib import Path
from threading import Event, Lock, Thread

from .base import GenerationAdapter
from ..config import settings
from ..media import LocalReference
from ..lora import download_hugging_face_lora, load_lora_adapter
from ..model_cache import missing_components, required_components, resolve_model_snapshot
from ..schemas import GenerationInput
from ..utils import aspect_dimensions, h3_num_frames


class DiffusersH3Adapter(GenerationAdapter):
    """Lazy official Modular Diffusers adapter. Importing this module never loads or downloads H3."""

    _pipelines: dict[str, object] = {}
    _attention_backends: dict[str, str] = {}
    _loaded_loras: dict[str, str] = {}
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

    @staticmethod
    def _configure_attention(pipeline, mode: str, torch) -> str:
        transformer_name = "transformer_ref" if mode == "ref2va" else "transformer"
        transformer = getattr(pipeline, transformer_name)
        configured = settings.attention_backend
        if configured == "auto":
            major, minor = torch.cuda.get_device_capability()
            if major >= 10:
                configured = "flash_4_hub"
            elif major == 9:
                configured = "_flash_3_hub"
            else:
                configured = "default"
            print(
                f"[h3] GPU={torch.cuda.get_device_name()} compute_capability={major}.{minor}; "
                f"attention_backend={configured}",
                flush=True,
            )
        if configured in {"", "default", "sdpa"}:
            print(
                "[h3] using the default attention backend; full-attention H3 generation may be slow",
                flush=True,
            )
            return "default"
        try:
            transformer.set_attention_backend(configured)
        except Exception as error:
            if settings.attention_backend != "auto":
                raise RuntimeError(
                    f"Unable to enable the requested H3 attention backend {configured}: {error}"
                ) from error
            print(
                f"[h3] WARNING: unable to enable {configured}: {error}; falling back to default attention",
                flush=True,
            )
            return "default"
        print(f"[h3] enabled attention backend {configured}", flush=True)
        return configured

    @staticmethod
    def _generate_with_heartbeat(pipeline, kwargs: dict[str, object], summary: str):
        stopped = Event()
        started = time.monotonic()

        def report() -> None:
            while not stopped.wait(30):
                elapsed = time.monotonic() - started
                print(
                    f"[h3] generation still active after {elapsed:.0f}s ({summary}); "
                    "the denoising progress bar advances after each full transformer step",
                    flush=True,
                )

        heartbeat = Thread(target=report, name="h3-generation-heartbeat", daemon=True)
        heartbeat.start()
        try:
            return pipeline(**kwargs)
        finally:
            stopped.set()
            heartbeat.join(timeout=1)

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
            load_kwargs: dict[str, object] = {
                "dtype": torch.bfloat16,
                "cache_dir": str(settings.hub_cache),
            }
            if snapshot is not None:
                # Component specs in the modular index retain the Hub repository ID.
                # Override it so a production worker uses only its prepared volume.
                load_kwargs.update(
                    pretrained_model_name_or_path=str(snapshot),
                    local_files_only=True,
                )
            pipeline.load_components(**load_kwargs)

            unloaded = [
                name for name in required_components(mode) if getattr(pipeline, name, None) is None
            ]
            if unloaded:
                raise RuntimeError(
                    f"Diffusers failed to load required H3 components for {mode}: {', '.join(unloaded)}. "
                    "Check the component-load error immediately above; the worker image may be missing "
                    "a required runtime dependency."
                )
            self._attention_backends[mode] = self._configure_attention(pipeline, mode, torch)
            transformer_name = "transformer_ref" if mode == "ref2va" else "transformer"
            active_transformer = getattr(pipeline, transformer_name)
            active_transformer.requires_grad_(False)
            active_transformer.eval()
            self._pipelines[mode] = pipeline
            print(f"[h3] {mode} components ready in {time.monotonic() - started:.1f}s", flush=True)
            return pipeline

    @classmethod
    def _configure_lora(cls, pipeline, request: GenerationInput) -> str:
        transformer_name = "transformer_ref" if request.mode == "ref2va" else "transformer"
        transformer = getattr(pipeline, transformer_name)
        selected = request.loras[0] if request.loras else None
        current = cls._loaded_loras.get(request.mode)
        if selected is None:
            if current:
                transformer.disable_lora()
            pipeline.scheduler.set_shift(12.0)
            pipeline.audio_scheduler.set_shift(3.0)
            return "none"

        signature = f"{selected.url}|{selected.alpha}"
        if current and current != signature:
            transformer.delete_adapters("default")
            cls._loaded_loras.pop(request.mode, None)
        if cls._loaded_loras.get(request.mode) != signature:
            print(f"[h3] caching linked LoRA {selected.url}", flush=True)
            lora_path = download_hugging_face_lora(str(selected.url), settings.hub_cache)
            load_lora_adapter(transformer, lora_path, selected.alpha, selected.weight)
            cls._loaded_loras[request.mode] = signature
        else:
            transformer.enable_lora()
            transformer.set_adapters("default", weights=selected.weight)
        pipeline.scheduler.set_shift(selected.video_shift)
        pipeline.audio_scheduler.set_shift(selected.audio_shift)
        return selected.name

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
        active_lora = self._configure_lora(pipeline, request)
        frames = h3_num_frames(request.target.duration_seconds, request.target.fps)
        width, height = aspect_dimensions(request.target.aspect_ratio, request.target.short_edge)
        kwargs: dict[str, object] = {
            "prompt": request.resolved_prompt,
            "num_frames": frames,
            "height": height,
            "width": width,
            # MiniMaxH3Scheduler counts sigma grid points including terminal zero.
            # N transformer evaluations therefore require N + 1 grid points.
            "num_inference_steps": request.inference_steps + 1,
            "generator": torch.Generator(device="cpu").manual_seed(request.seed),
            "output_type": "np",
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

        summary = (
            f"mode={request.mode}, canvas={width}x{height}, frames={frames}, "
            f"steps={request.inference_steps}, attention={self._attention_backends.get(request.mode, 'unknown')}, "
            f"lora={active_lora}"
        )
        print(f"[h3] generation compute starting ({summary})", flush=True)
        generation_started = time.monotonic()
        with torch.inference_mode():
            result = self._generate_with_heartbeat(pipeline, kwargs, summary)
        print(
            f"[h3] generation compute completed in {time.monotonic() - generation_started:.1f}s; encoding output",
            flush=True,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio = result["audio"][0]
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio)
        audio = audio.detach()
        encode_video(
            result["videos"][0],
            fps=request.target.fps,
            output_path=str(output_path),
            audio=audio,
            audio_sample_rate=result["sampling_rate"],
        )
        print(f"[h3] output encoded at {output_path}", flush=True)
        return frames / request.target.fps
