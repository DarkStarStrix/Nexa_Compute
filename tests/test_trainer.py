from nexa_compute.config import load_config
from nexa_compute.data import DataPipeline
from nexa_compute.models import DEFAULT_MODEL_REGISTRY
from nexa_compute.training import Trainer


def test_trainer_runs_single_epoch() -> None:
    config = load_config("nexa_train/configs/baseline.yaml")
    config.training.epochs = 1
    config.training.max_steps = 2
    config.data.batch_size = 16
    pipeline = DataPipeline(config.data)
    train_loader = pipeline.dataloader("train")
    val_loader = pipeline.dataloader("validation")
    model = DEFAULT_MODEL_REGISTRY.build(config.model)
    trainer = Trainer(config, callbacks=[])
    state = trainer.fit(model, train_loader, val_loader)
    assert state.epoch == 0
    assert state.global_step >= 1
