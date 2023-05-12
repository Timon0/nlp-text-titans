import json

import pytorch_lightning as pl
import pytorch_lightning.loggers
import torch
import wandb
import numpy as np
import scoring
from datasets import load_dataset
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

class DataModule(pl.LightningDataModule):
    def  __init__(self, model_name, batch_size: int = 32):
        super().__init__()
        self.test_dataset = None
        self.validation_dataset = None
        self.train_dataset = None
        self.datasets = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # make sure the tokenizer truncates the beginning of the input, not the end
        self.tokenizer.padding_side = "left"
        self.batch_size = batch_size

    def prepare_data(self):
        # load data
        self.datasets = load_dataset('csv', data_files={'train': './data/cleaned_with_context.csv',
                                                        'test': './data/cleaned_with_context_test.csv'})

        # tokenize
        self.datasets = self.datasets.map(self.tokenize_data, batched=True)

        # remove unused columns
        self.datasets = self.datasets.remove_columns(['humor', 'context', 'target'])

        # set correct format
        self.datasets.set_format(type="torch")

    def tokenize_data(self, datasets, padding = "max_length"):
        

        # tokenize inputs
        model_inputs = self.tokenizer(list(map(str, datasets['context'])), max_length=512, padding=padding, truncation=True, return_tensors="pt")

        # Tokenize targets with the `text_target` keyword argument
        labels = self.tokenizer(text_target=list(map(str, datasets['target'])), max_length=512, padding=padding, truncation=True, return_tensors="pt")

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
    def __init__(self, model_name, rank: int = 16):
        super().__init__()

        self.save_hyperparameters()

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )        
        native_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
        self.model = get_peft_model(native_model, lora_config)

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
        predictions = outputs.logits
        # predictions = torch.argmax(logits, dim=-1)
        # accuracy = (labels == predictions).float().mean()

        # compute f1 and exact match
        #f1 = self.compute_f1(self.tokenizer.batch_decode(predictions), self.tokenizer.batch_decode(labels))


        self.log('val_loss', outputs.loss)
        #self.log('val_f1', f1)
        # the validation_step method expects a dictionary, which should at least contain the val_loss
        return {'val_loss': outputs.loss}

    def compute_f1(predictions, labels):
        f1_scores = []
        for prediction, label in zip(predictions, labels):
            f1_scores.append(scoring.compute_f1(prediction, label))
        return np.array(f1_scores).mean()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-5)

        return optimizer


if __name__ == "__main__":
    torch.cuda.empty_cache()
    seed_everything(42, workers=True)
    
    # Hyperparams
    model_name = "google/flan-t5-base"
    batch_size = 8
    lora_rank = 8

    # Logger
    with open("./config/config.json", "r") as jsonfile:
        data = json.load(jsonfile)
        subscription_key = data['wandb']['subscription_key']

    wandb.login(key=subscription_key)
    wandb_logger = pytorch_lightning.loggers.WandbLogger(project="text-titans-lora-flan")
    
    # Training
    data_module = DataModule(model_name, batch_size)
    model = Model(model_name, lora_rank)
    trainer = Trainer(logger=wandb_logger, max_epochs=5)
    trainer.fit(model, datamodule=data_module)
