FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . ./
ENV PYTHONPATH="/workspace/src"

CMD ["python", "scripts/cli.py", "train", "--config", "configs/default.yaml"]
