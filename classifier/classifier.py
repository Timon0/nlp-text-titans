import json

import pytorch_lightning as pl
import pytorch_lightning.loggers
import torch
import wandb
from datasets import load_dataset
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
from torch.optim import AdamW
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def prepare_labels(dataset):
    humor = dataset['humor']
    label = 1 if humor > 0 else 0
    return {'labels': label}


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
        self.datasets = load_dataset('csv', data_files={'train': './data/oasst1_train_en_cleaned.csv',
                                                        'test': './data/oasst1_test_en_cleaned.csv'})
        # prepare labels
        self.datasets = self.datasets.map(prepare_labels)

        # tokenize
        self.datasets = self.datasets.map(self.tokenize_data, batched=True)

        # remove unused columns
        self.datasets = self.datasets.remove_columns(['message_id', 'role', 'lang', 'humor', 'text'])

        # set correct format
        self.datasets.set_format(type="torch")

    def tokenize_data(self, datasets):
        return self.tokenizer(datasets['text'], padding="max_length", truncation=True)

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
    def __init__(self, model_name, batch_size, learning_rate):
        super().__init__()

        self.save_hyperparameters()

        self.classifier = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def forward(self, batch):
        return self.classifier(**batch)

    def training_step(self, batch, batch_nb):
        outputs = self(batch)

        self.log('train_loss', outputs.loss)

        return {'loss': outputs.loss}

    def validation_step(self, batch, batch_nb):
        outputs = self(batch)

        # Apart from the validation loss, we also want to track validation accuracy to get an idea, what the
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
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self._num_steps() * 0.1,
            num_training_steps=self._num_steps(),
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def _num_steps(self) -> int:
        """Get number of steps"""
        train_dataloader = self.trainer.datamodule.train_dataloader()
        dataset_size = len(train_dataloader.dataset)
        num_steps = dataset_size * self.trainer.max_epochs // self.batch_size
        return num_steps


# %%
if __name__ == "__main__":

    seed_everything(42, workers=True)

    # Hyperparams
    model_name = "distilbert-base-cased"
    batch_size = 32
    learning_rate = 2e-5

    # Logger
    with open("./config/config.json", "r") as jsonfile:
        data = json.load(jsonfile)
        subscription_key = data['wandb']['subscription_key']

    wandb.login(key=subscription_key)
    wandb_logger = pytorch_lightning.loggers.WandbLogger(project="text-titans")

    # Training
    data_module = DataModule(model_name, batch_size)
    model = Model(model_name, batch_size, learning_rate)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(logger=wandb_logger, max_epochs=5, callbacks=[lr_monitor])
    trainer.fit(model, datamodule=data_module)
