from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from h3_worker.model_cache import missing_components


MODEL_ID = "MiniMaxAI/MiniMax-H3"
COMMON_PATTERNS = [
    "model_index.json",
    "modular_model_index.json",
    "processor/*",
    "tokenizer/*",
    "text_encoder/*",
    "vae/*",
    "audio_vae/*",
    "scheduler/*",
    "audio_scheduler/*",
]
ESTIMATED_GIB = {"base": 134.16, "ref": 134.16, "both": 195.89}
RECOMMENDED_FREE_GIB = {"base": 150, "ref": 150, "both": 215}


def parse_args() -> argparse.Namespace:
    hf_home = Path(os.getenv("HF_HOME", "/workspace/huggingface"))
    parser = argparse.ArgumentParser(description="Preload only the required MiniMax H3 Diffusers files.")
    parser.add_argument("--workflow", choices=("base", "ref", "both"), required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.getenv("HF_HUB_CACHE", str(hf_home / "hub"))),
        help="Hugging Face hub cache on the attached network volume.",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--force-low-space", action="store_true")
    return parser.parse_args()


def patterns_for(workflow: str) -> list[str]:
    patterns = list(COMMON_PATTERNS)
    if workflow in {"base", "both"}:
        patterns.append("transformer/*")
    if workflow in {"ref", "both"}:
        patterns.append("transformer_ref/*")
    return patterns


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.cache_dir.parent))
    os.environ.setdefault("HF_HUB_CACHE", str(args.cache_dir))
    os.environ.setdefault("HF_XET_CACHE", str(args.cache_dir.parent / "xet"))
    from huggingface_hub import snapshot_download

    free_gib = shutil.disk_usage(args.cache_dir).free / 1024**3
    required = RECOMMENDED_FREE_GIB[args.workflow]
    print(
        f"Preparing {args.workflow} cache: approximately {ESTIMATED_GIB[args.workflow]:.2f} GiB "
        f"of model files; {free_gib:.1f} GiB currently free.",
        flush=True,
    )
    if free_gib < required and not args.force_low_space:
        raise SystemExit(
            f"At least {required} GiB free is recommended. Resize the volume or pass --force-low-space."
        )

    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=args.revision,
            cache_dir=str(args.cache_dir),
            allow_patterns=patterns_for(args.workflow),
            token=os.getenv("HF_TOKEN"),
        )
    )
    modes = ["t2va", "fl2va"] if args.workflow == "base" else ["ref2va"]
    if args.workflow == "both":
        modes = ["t2va", "fl2va", "ref2va"]
    for mode in modes:
        missing = missing_components(snapshot, mode)
        if missing:
            raise SystemExit(f"Cache validation failed for {mode}; missing: {', '.join(missing)}")
    print(f"H3 {args.workflow} cache is ready at {snapshot}", flush=True)


if __name__ == "__main__":
    main()
