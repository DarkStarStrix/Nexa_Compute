from pathlib import Path

from nexa_compute.config import load_config


def test_load_default_config(tmp_path: Path) -> None:
    config = load_config(Path("nexa_train/configs/baseline.yaml"))
    assert config.data.batch_size == 64
    assert config.model.name == "mlp_classifier"
    assert config.training.optimizer.lr == 0.0005
