FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG PYTHON_VERSION=3.11

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    tmux \
    libaio1 \
    openssh-client \
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

RUN pip install flash-attn --no-build-isolation || true

RUN pip install \
    "deepspeed==0.14.4" \
    "ninja==1.11.1.1" \
    "bitsandbytes==0.48.2"

RUN pip install --no-deps "axolotl==0.12.2"

RUN pip install mpi4py "xformers==0.0.27.post2" --extra-index-url https://download.pytorch.org/whl/cu121 || true

RUN useradd -ms /bin/bash runner && chown -R runner:runner /workspace
USER runner

ENV PATH="/home/runner/.local/bin:${PATH}"
ENV HF_HOME=/home/runner/.cache/huggingface

ENTRYPOINT ["/bin/bash"]

