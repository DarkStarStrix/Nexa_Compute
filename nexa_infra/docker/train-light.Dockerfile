FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG PYTHON_VERSION=3.11

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    tmux \
    libaio1 \
    wget \
    "python${PYTHON_VERSION}" \
    python3-pip && \
    ln -s "/usr/bin/python${PYTHON_VERSION}" /usr/bin/python && \
    pip3 install --upgrade pip uv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

RUN pip install \
    "transformers==4.*" \
    "datasets==2.*" \
    "accelerate>=0.30" \
    "peft>=0.11" \
    "bitsandbytes>=0.43" \
    "trl>=0.9" \
    sentencepiece \
    wandb \
    evaluate \
    scikit-learn

RUN useradd -ms /bin/bash runner && chown -R runner:runner /workspace
USER runner

ENV PATH="/home/runner/.local/bin:${PATH}"
ENV HF_HOME=/home/runner/.cache/huggingface

ENTRYPOINT ["/bin/bash"]

