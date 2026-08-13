from __future__ import annotations

from pathlib import Path


RUNPOD_CACHED_MODEL_HUB = Path("/runpod-volume/huggingface-cache/hub")
SHARED_COMPONENTS = (
    "processor",
    "tokenizer",
    "text_encoder",
    "vae",
    "audio_vae",
    "scheduler",
    "audio_scheduler",
)


def repository_cache_name(model_id: str) -> str:
    return f"models--{model_id.replace('/', '--')}"


def _valid_snapshot(path: Path) -> bool:
    return path.is_dir() and any(
        (path / index).is_file() for index in ("modular_model_index.json", "model_index.json")
    )


def snapshot_from_hub_cache(model_id: str, hub_cache: Path) -> Path | None:
    model_root = hub_cache / repository_cache_name(model_id)
    snapshots = model_root / "snapshots"
    main_ref = model_root / "refs" / "main"

    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
        candidate = snapshots / revision
        if _valid_snapshot(candidate):
            return candidate

    if not snapshots.is_dir():
        return None
    candidates = [path for path in snapshots.iterdir() if _valid_snapshot(path)]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def resolve_model_snapshot(
    model_id: str,
    hub_cache: Path,
    explicit_path: Path | None = None,
) -> Path | None:
    if explicit_path is not None:
        expanded = explicit_path.expanduser()
        if not _valid_snapshot(expanded):
            raise RuntimeError(
                f"H3_MODEL_PATH does not contain modular_model_index.json or model_index.json: {expanded}"
            )
        return expanded

    cache_roots = [hub_cache]
    if RUNPOD_CACHED_MODEL_HUB not in cache_roots:
        cache_roots.append(RUNPOD_CACHED_MODEL_HUB)
    for cache_root in cache_roots:
        snapshot = snapshot_from_hub_cache(model_id, cache_root)
        if snapshot is not None:
            return snapshot
    return None


def required_components(mode: str) -> list[str]:
    transformer = "transformer_ref" if mode == "ref2va" else "transformer"
    return [*SHARED_COMPONENTS, transformer]


def missing_components(snapshot: Path, mode: str) -> list[str]:
    return [name for name in required_components(mode) if not (snapshot / name).is_dir()]
