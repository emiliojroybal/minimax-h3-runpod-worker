# MiniMax H3 RunPod deployment

The public H3 repository is too large to enter directly in RunPod's **Model** field. That field caches the entire repository (about 464 GiB), including duplicate original and Diffusers layouts. This worker instead uses a persistent network volume containing only the files required by each endpoint.

## Recommended layout

Use two Serverless endpoints:

| Endpoint | Modes | Model data | Suggested volume |
| --- | --- | ---: | ---: |
| Base | `t2va`, `fl2va` | about 134 GiB | 200 GB or larger |
| Reference | `ref2va` | about 134 GiB | 200 GB or larger |

This keeps a worker from retaining both large transformer variants. A single combined endpoint is supported, but its volume should be at least 300 GB and it may require more host memory.

## 1. Prepare the Base network volume

1. Create a 200 GB or larger RunPod network volume.
2. Create a temporary Pod in the same data center and attach that volume. It appears as `/workspace` inside the Pod.
3. In the Pod terminal, run:

```bash
git clone https://github.com/emiliojroybal/minimax-h3-runpod-worker.git
cd minimax-h3-runpod-worker
python3 -m pip install --upgrade "huggingface-hub[hf-xet]==1.27.0"
export HF_HOME=/workspace/huggingface
export HF_HUB_CACHE=/workspace/huggingface/hub
export HF_XET_CACHE=/workspace/huggingface/xet
python3 preload_h3.py --workflow base --cache-dir /workspace/huggingface/hub
```

The preload downloads approximately 134 GiB once and validates the resulting snapshot. It does not load the model or require a GPU. Stop the temporary Pod after it reports that the cache is ready; keep the network volume.

## 2. Prepare the Ref2VA network volume

Repeat the preceding process with a second 200 GB volume and this command:

```bash
python3 preload_h3.py --workflow ref --cache-dir /workspace/huggingface/hub
```

For a single combined volume instead, use:

```bash
python3 preload_h3.py --workflow both --cache-dir /workspace/huggingface/hub
```

## 3. Configure the Base endpoint

Create or edit the Serverless endpoint:

- Leave RunPod's **Model** field empty.
- Attach the prepared Base network volume.
- Select the worker image built from this directory.
- Use a GPU with at least 80 GB VRAM and a host with substantial system RAM. The adapter uses CPU offloading.
- Set the following environment variables:

```text
H3_INFERENCE_MODE=diffusers
H3_MODEL_ID=MiniMaxAI/MiniMax-H3
H3_REQUIRE_LOCAL_MODEL=true
H3_ATTENTION_BACKEND=auto
H3_ALLOWED_MODES=t2va,fl2va
HF_HOME=/runpod-volume/huggingface
HF_HUB_CACHE=/runpod-volume/huggingface/hub
HF_XET_CACHE=/runpod-volume/huggingface/xet
H3_OUTPUT_DIR=/runpod-volume/outputs
```

Also configure the `OBJECT_STORAGE_*` values from `.env.example` for reference inputs and downloadable output.

## 4. Configure the Ref2VA endpoint

Use the same settings with the Ref2VA volume and change:

```text
H3_ALLOWED_MODES=ref2va
```

## 5. Configure the local Studio

On the local Studio server, set:

```text
INFERENCE_BACKEND=runpod
RUNPOD_API_KEY=YOUR_RUNPOD_API_KEY
RUNPOD_BASE_ENDPOINT_ID=YOUR_BASE_ENDPOINT_ID
RUNPOD_REF_ENDPOINT_ID=YOUR_REFERENCE_ENDPOINT_ID
```

The Studio records the selected endpoint together with each RunPod job ID, so status checks and cancellation continue to work across both endpoints. `RUNPOD_ENDPOINT_ID` remains available as a single-endpoint fallback.

## 6. Verify before generating

Send this async request to each endpoint:

```json
{
  "input": {
    "operation": "capabilities"
  }
}
```

The response should report `model_cache.ready: true` and list the endpoint's enabled modes. This health request does not load model weights into memory. The first actual generation will still take several minutes to load the pre-cached weights from the volume, but it should not download them from Hugging Face.

During the Docker build, look for a `PyTorch runtime ready` line. This confirms that the matching Torch, Torchvision, and Torchaudio wheels are installed and that the Qwen3-VL video processor can be imported. If an older worker reports `Qwen3VLVideoProcessor requires the Torchvision library`, rebuild and redeploy the worker image from the latest commit; the prepared model volume does not need to be downloaded again.

On an H100 or H200, the first generation should log `enabled attention backend _flash_3_hub`. The small precompiled kernel is fetched from Hugging Face and cached under `HF_HOME`; it is not part of the H3 model weights. If auto-selection cannot enable it, the worker logs a warning and falls back to the much slower full-attention path. During generation, a heartbeat is logged every 30 seconds even while a single transformer step is still running.

## Hardware warning

This worker uses Diffusers automatic CPU offloading on one GPU. The open BF16 H3 checkpoint remains extremely large. The official MiniMax SGLang examples use four GPUs. If an 80 GB Serverless worker runs out of GPU or host memory, use a multi-GPU SGLang deployment or a separately validated quantized checkpoint; increasing cache storage alone will not solve an out-of-memory error.
