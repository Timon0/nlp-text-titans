import json

import pytorch_lightning.loggers
import torch
import wandb
from pytorch_lightning import Trainer, seed_everything

from model import DataModule, Model

if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    seed_everything(42, workers=True)

    # Params
    checkpoint_path = "checkpoints/google/flan-t5-base-batch4.ckpt"

    # Logger
    with open("./config/config.json", "r") as jsonfile:
        data = json.load(jsonfile)
        subscription_key = data['wandb']['subscription_key']

    wandb.login(key=subscription_key)
    wandb_logger = pytorch_lightning.loggers.WandbLogger(project="text-titans-lora-flan")

    # Load Model
    model = Model.load_from_checkpoint(checkpoint_path)
    data_module = DataModule(model.hparams.model_name, model.hparams.batch_size)
    trainer = Trainer(logger=wandb_logger)

    # Test
    trainer.test(model, datamodule=data_module)
