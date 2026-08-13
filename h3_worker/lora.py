from __future__ import annotations

import gc
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import unquote, urlparse


LORA_TARGET_MODULES = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)
LORA_A_SUFFIX = ".lora_A.default.weight"
LORA_B_SUFFIX = ".lora_B.default.weight"
SUPPORTED_REPO = "lightx2v/Minimax-h3-Turbo"
SUPPORTED_FILENAMES = {
    "minimax_h3_fl2v_turbo_4step_v0.1.safetensors",
    "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors",
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
    "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
}

# The pure-PEFT validation/loading contract follows ModelTC/Minimax-H3-Turbo's
# Apache-2.0 Diffusers inference implementation.


def parse_hugging_face_lora_url(url: str) -> tuple[str, str, str]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
        raise ValueError("LoRA links must use https://huggingface.co.")
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 5 or parts[2] not in {"blob", "resolve"}:
        raise ValueError(
            "Use a Hugging Face file link ending in /blob/<revision>/<file>.safetensors "
            "or /resolve/<revision>/<file>.safetensors."
        )
    repo_id = "/".join(parts[:2])
    revision = parts[3]
    filename = "/".join(parts[4:])
    if not filename.lower().endswith(".safetensors"):
        raise ValueError("The LoRA link must point to a .safetensors file.")
    if repo_id.lower() != SUPPORTED_REPO.lower() or filename.lower() not in SUPPORTED_FILENAMES:
        raise ValueError("Only supported lightx2v/Minimax-h3-Turbo Diffusers checkpoints are accepted.")
    return repo_id, revision, filename


def download_hugging_face_lora(url: str, cache_dir: Path) -> Path:
    """Resolve a public Hugging Face file link into the persistent Hub cache."""
    repo_id, revision, filename = parse_hugging_face_lora_url(url)

    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type="model",
            cache_dir=str(cache_dir),
        )
    )


def _load_lora_state_dict(path: Path) -> Mapping[str, object]:
    from safetensors.torch import load_file

    checkpoint = load_file(path, device="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Expected a state-dict mapping in {path}.")
    return checkpoint


def _validate_lora_state_dict(state_dict: Mapping[str, object], path: Path) -> int:
    import torch

    lora_a: dict[str, torch.Tensor] = {}
    lora_b: dict[str, torch.Tensor] = {}
    unsupported: list[str] = []
    for key, value in state_dict.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError(f"LoRA checkpoint contains invalid entries: {path}")
        if key.endswith(LORA_A_SUFFIX):
            lora_a[key[: -len(LORA_A_SUFFIX)]] = value
        elif key.endswith(LORA_B_SUFFIX):
            lora_b[key[: -len(LORA_B_SUFFIX)]] = value
        else:
            unsupported.append(key)
    if unsupported:
        raise ValueError(
            f"{path} is not a pure MiniMax H3 PEFT LoRA; unsupported keys: {', '.join(unsupported[:3])}"
        )
    if not lora_a:
        raise ValueError(f"No MiniMax H3 LoRA tensors were found in {path}.")
    if lora_a.keys() != lora_b.keys():
        raise ValueError("The LoRA contains unpaired A/B tensors.")

    ranks: set[int] = set()
    for module_name, a_tensor in lora_a.items():
        b_tensor = lora_b[module_name]
        if a_tensor.ndim != 2 or b_tensor.ndim != 2 or a_tensor.shape[0] != b_tensor.shape[1]:
            raise ValueError(f"Invalid LoRA tensor shapes for {module_name}.")
        if not module_name.endswith(LORA_TARGET_MODULES):
            raise ValueError(f"Unsupported MiniMax H3 LoRA target module {module_name!r}.")
        ranks.add(a_tensor.shape[0])
    if len(ranks) != 1:
        raise ValueError(f"Mixed LoRA ranks are unsupported: {sorted(ranks)}")
    return ranks.pop()


def load_lora_adapter(transformer, path: Path, alpha: int, weight: float) -> None:
    """Load ModelTC's pure-PEFT H3 LoRA format into the active transformer."""
    from peft import LoraConfig

    state_dict = _load_lora_state_dict(path)
    rank = _validate_lora_state_dict(state_dict, path)
    transformer.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights=False,
            target_modules=list(LORA_TARGET_MODULES),
            use_rslora=False,
        ),
        adapter_name="default",
    )
    adapter_parameters = {
        name: parameter
        for name, parameter in transformer.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    missing = sorted(adapter_parameters.keys() - state_dict.keys())
    unexpected = sorted(state_dict.keys() - adapter_parameters.keys())
    mismatched = [
        name
        for name, parameter in adapter_parameters.items()
        if name in state_dict and state_dict[name].shape != parameter.shape
    ]
    if missing or unexpected or mismatched:
        transformer.delete_adapters("default")
        raise ValueError(
            "LoRA is incompatible with the loaded H3 transformer: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}, shape_mismatches={mismatched[:3]}."
        )
    incompatible = transformer.load_state_dict(state_dict, strict=False)
    missing_lora = [
        key for key in incompatible.missing_keys if ".lora_A." in key or ".lora_B." in key
    ]
    if incompatible.unexpected_keys or missing_lora:
        transformer.delete_adapters("default")
        raise RuntimeError(
            "LoRA loading did not consume the expected tensors: "
            f"missing={missing_lora[:3]}, unexpected={incompatible.unexpected_keys[:3]}."
        )
    transformer.set_adapters("default", weights=weight)
    transformer.requires_grad_(False)
    transformer.eval()
    tensor_count = len(state_dict)
    del state_dict
    gc.collect()
    print(
        f"[h3] loaded LoRA path={path} tensors={tensor_count} rank={rank} "
        f"alpha={alpha} weight={weight} effective_scale={weight * alpha / rank:.8g}",
        flush=True,
    )
