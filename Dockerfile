FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    H3_INFERENCE_MODE=diffusers \
    HF_HOME=/runpod-volume/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip git ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-mock.txt requirements-production.txt ./
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 torch \
    && python3 -m pip install -r requirements-production.txt
COPY h3_worker ./h3_worker
COPY handler.py .

# Deliberately no model download here. H3 components are fetched by load_components()
# only after a production request reaches a suitably provisioned GPU worker.
CMD ["python3", "-u", "handler.py"]
