import json

import pytorch_lightning as pl
import pytorch_lightning.loggers
import torch
import wandb
import numpy as np
from datasets import load_dataset
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class DataModule(pl.LightningDataModule):
    def  __init__(self, model_name, batch_size: int = 32):
        super().__init__()
        self.test_dataset = None
        self.validation_dataset = None
        self.train_dataset = None
        self.datasets = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size

    def prepare_data(self):
        # load data
        # TODO: Check filenames so they match!!
        self.datasets = load_dataset('csv', data_files={'train': './data/oasst1_train_cleaned.csv',
                                                        'test': './data/oasst1_test_cleaned.csv'})

        # tokenize
        self.datasets = self.datasets.map(self.tokenize_data, batched=True)

        # remove unused columns
        self.datasets = self.datasets.remove_columns(['humor', 'prompt', 'target'])

        # set correct format
        self.datasets.set_format(type="torch")

    def tokenize_data(self, datasets, padding = "max_length"):
        # The maximum total input sequence length after tokenization. 
        # Sequences longer than this will be truncated, sequences shorter will be padded.
        tokenized_inputs = datasets.map(lambda x: self.tokenizer(x["prompt"], truncation=True), batched=True)
        input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
        # take 85 percentile of max length for better utilization
        max_source_length = int(np.percentile(input_lenghts, 85))

        # The maximum total sequence length for target text after tokenization. 
        # Sequences longer than this will be truncated, sequences shorter will be padded."
        tokenized_targets = datasets.map(lambda x: self.tokenizer(x["text"], truncation=True), batched=True)
        target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
        # take 90 percentile of max length for better utilization
        max_target_length = int(np.percentile(target_lenghts, 90))

        inputs = datasets["context"]

        # tokenize inputs
        model_inputs = self.tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True, return_tensors="pt")

        # Tokenize targets with the `text_target` keyword argument
        labels = self.tokenizer(text_target=datasets['target'], max_length=max_target_length, padding=padding, truncation=True, return_tensors="pt")

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def setup(self, stage: str):
        # split data
        train_validation_dataset = self.datasets['train'].train_test_split(test_size=0.1)
        self.train_dataset = train_validation_dataset['train']
        self.validation_dataset = train_validation_dataset['test']
        self.test_dataset = self.datasets['test']

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        pass


class Model(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()

        self.save_hyperparameters()

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_nb):
        outputs = self(batch)

        self.log('train_loss', outputs.loss)

        return {'loss': outputs.loss}

    def validation_step(self, batch, batch_nb):
        outputs = self(batch)

        # Apart from the validation loss, we also want to track validation accuracy  to get an idea, what the
        # model training has achieved "in real terms".
        labels = batch['labels']
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (labels == predictions).float().mean()

        self.log('val_loss', outputs.loss)
        self.log('val_accuracy', accuracy)
        # the validation_step method expects a dictionary, which should at least contain the val_loss
        return {'val_loss': outputs.loss, 'val_accuracy': accuracy}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-5)

        return optimizer


# %%
if __name__ == "__main__":

    seed_everything(42, workers=True)

    # Hyperparams
    model_name = "google/flan-t5-small"
    batch_size = 32

    # Logger
    with open("./config/config.json", "r") as jsonfile:
        data = json.load(jsonfile)
        subscription_key = data['wandb']['subscription_key']

    wandb.login(key=subscription_key)
    wandb_logger = pytorch_lightning.loggers.WandbLogger(project="text-titans-2")

    # Training
    data_module = DataModule(model_name, batch_size)
    model = Model(model_name)
    trainer = Trainer(logger=wandb_logger, max_epochs=5)
    trainer.fit(model, datamodule=data_module)
