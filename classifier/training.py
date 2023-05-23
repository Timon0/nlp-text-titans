import json

import pytorch_lightning.loggers
import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from model import DataModule, Model


if __name__ == "__main__":
    torch.cuda.empty_cache()
    seed_everything(42, workers=True)

    # Hyperparams
    model_name = "distilbert-base-multilingual-cased"
    batch_size = 32
    learning_rate = 2e-5

    # Logger
    with open("./config/config.json", "r") as jsonfile:
        data = json.load(jsonfile)
        subscription_key = data["wandb"]["subscription_key"]

    wandb.login(key=subscription_key)
    wandb_logger = pytorch_lightning.loggers.WandbLogger(project="text-titans")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"{model_name}-batch{batch_size}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Training
    data_module = DataModule(model_name, batch_size)
    model = Model(model_name, batch_size, learning_rate)
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=5,
        precision=16,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor]
    )
    trainer.fit(model, datamodule=data_module)
