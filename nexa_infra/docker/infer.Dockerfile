FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG PYTHON_VERSION=3.11

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    tmux \
    "python${PYTHON_VERSION}" \
    python3-pip && \
    ln -s "/usr/bin/python${PYTHON_VERSION}" /usr/bin/python && \
    pip3 install --upgrade pip uv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN pip install --index-url https://download.pytorch.org/whl/cu121 torch

RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

RUN pip install \
    "vllm==0.*" \
    "transformers==4.*" \
    fastapi \
    uvicorn \
    pydantic

COPY docker/entrypoints/vllm_entry.sh /usr/local/bin/vllm_entry.sh
RUN chmod +x /usr/local/bin/vllm_entry.sh

RUN useradd -ms /bin/bash runner && chown -R runner:runner /workspace
USER runner

ENV PATH="/home/runner/.local/bin:${PATH}"
ENV HF_HOME=/home/runner/.cache/huggingface

ENTRYPOINT ["/usr/local/bin/vllm_entry.sh"]

