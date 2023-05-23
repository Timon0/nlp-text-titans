import json

import pytorch_lightning.loggers
import torch
import wandb
from pytorch_lightning import Trainer, seed_everything

from model import DataModule, Model

if __name__ == "__main__":
    torch.cuda.empty_cache()
    seed_everything(42, workers=True)

    # Params
    baseline_models = ["bert-base-multilingual-cased", "distilbert-base-multilingual-cased"]
    model_name = baseline_models[0]
    batch_size = 16

    # Logger
    with open("./config/config.json", "r") as jsonfile:
        data = json.load(jsonfile)
        subscription_key = data["wandb"]["subscription_key"]
    wandb.login(key=subscription_key)
    wandb_logger = pytorch_lightning.loggers.WandbLogger(project="text-titans")

    # Baseline
    data_module = DataModule(model_name, batch_size)
    baseline = Model(model_name, batch_size)
    trainer = Trainer(logger=wandb_logger, precision=16)

    # Validation
    trainer.validate(baseline, datamodule=data_module)
    # Testing
    trainer.test(baseline, datamodule=data_module)
