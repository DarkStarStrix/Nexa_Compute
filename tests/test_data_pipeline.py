import torch

from nexa_compute.config import load_config
from nexa_compute.data import DataPipeline


def test_data_pipeline_builds_dataloader() -> None:
    config = load_config("nexa_train/configs/baseline.yaml")
    pipeline = DataPipeline(config.data)
    loader = pipeline.dataloader("train")
    batch = next(iter(loader))
    features, labels = batch
    assert isinstance(features, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert features.shape[0] == config.data.batch_size
