import json

import pytorch_lightning.loggers
import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from model import DataModule, Model

if __name__ == "__main__":
    torch.cuda.empty_cache()
    seed_everything(42, workers=True)

    # Params
    baseline_models = [("google/flan-t5-small", 12), ("google/flan-t5-base", 4), ("google/flan-t5-large", 1)]
    learning_rate = 1e-3
    # Loop through baseline models
    for (model_name, batch_size) in baseline_models:
        # Logger
        with open("./config/config.json", "r") as jsonfile:
            data = json.load(jsonfile)
            subscription_key = data['wandb']['subscription_key']

        wandb.login(key=subscription_key)
        wandb_logger = pytorch_lightning.loggers.WandbLogger(project="text-titans-lora-flan")

        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints',
            filename=f"{model_name}-batch{batch_size}",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min"
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # Training
        data_module = DataModule(model_name, batch_size)
        model = Model(model_name, batch_size, learning_rate)
        # model = Model.load_from_checkpoint(checkpoint_path="checkpoint/epoch=4-step=26440.ckpt")

        trainer = Trainer(
            logger=wandb_logger,
            max_epochs=5,
            callbacks=[checkpoint_callback],
            accumulate_grad_batches=5
        )

        # Validation
        trainer.validate(model, datamodule=data_module)
        # Testing
        trainer.test(model, datamodule=data_module)
