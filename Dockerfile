FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    H3_INFERENCE_MODE=diffusers \
    H3_REQUIRE_LOCAL_MODEL=1 \
    HF_HOME=/runpod-volume/huggingface \
    HF_HUB_CACHE=/runpod-volume/huggingface/hub \
    HF_XET_CACHE=/runpod-volume/huggingface/xet

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip git ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-mock.txt requirements-production.txt ./
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 \
    && python3 -m pip install -r requirements-production.txt \
    && python3 -m pip check
COPY h3_worker ./h3_worker
COPY handler.py .
COPY preload_h3.py .

# Deliberately no model download here. Production requires a selectively preloaded
# network volume, preventing accidental 100+ GiB downloads during a request.
CMD ["python3", "-u", "handler.py"]
