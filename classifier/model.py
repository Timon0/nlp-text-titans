import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
import os


def prepare_labels(dataset):
    humor = dataset['humor']
    label = 1 if humor > 0.3 else 0
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
        dirname = os.path.dirname(__file__)
        self.datasets = load_dataset('csv',
                                     data_files={'train': os.path.join(dirname, './data/oasst1_train_cleaned.csv'),
                                                 'test': os.path.join(dirname, './data/oasst1_test_cleaned.csv')})

        # prepare labels
        self.datasets = self.datasets.map(prepare_labels)

        # tokenize
        self.datasets = self.datasets.map(self.tokenize_data, batched=True)

        # remove unused columns
        self.datasets = self.datasets.remove_columns(['message_id', 'role', 'lang', 'humor', 'text', 'review_count'])

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
    def __init__(self, model_name, batch_size, learning_rate=2e-5):
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
        return {'val_loss': outputs.loss, 'val_accuracy': accuracy}

    def test_step(self, batch, batch_nb):
        outputs = self(batch)

        # Apart from the test loss, we also want to track test accuracy to get an idea, what the
        # model training has achieved "in real terms".
        labels = batch['labels']
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (labels == predictions).float().mean()

        self.log('test_loss', outputs.loss)
        self.log('test_accuracy', accuracy)
        return {'test_loss': outputs.loss, 'test_accuracy': accuracy}

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