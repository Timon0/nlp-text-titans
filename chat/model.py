import pytorch_lightning as pl
import evaluate
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from sklearn.model_selection import train_test_split


class DataModule(pl.LightningDataModule):
    def __init__(self, model_name, batch_size: int = 32):
        super().__init__()
        self.test_dataset = None
        self.validation_dataset = None
        self.train_dataset = None
        self.datasets = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side='left')
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

    def tokenize_data(self, datasets, padding="max_length"):
        # tokenize inputs
        model_inputs = self.tokenizer(list(map(str, datasets['context'])), max_length=512, padding=padding,
                                      truncation=True, return_tensors="pt")

        # Tokenize targets with the `text_target` keyword argument
        labels = self.tokenizer(text_target=list(map(str, datasets['target'])), max_length=512, padding=padding,
                                truncation=True, return_tensors="pt")

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
        message_tree_ids = self.datasets['train']['message_tree_id']
        train_ids, val_ids = train_test_split(list(set(message_tree_ids)), test_size=0.1)

        self.train_dataset = self.datasets['train'].filter(lambda sample: sample['message_tree_id'] in train_ids)
        self.validation_dataset = self.datasets['train'].filter(lambda sample: sample['message_tree_id'] in val_ids)
        self.test_dataset = self.datasets['test']

        self.train_dataset = self.train_dataset.remove_columns(['message_tree_id'])
        self.validation_dataset = self.validation_dataset.remove_columns(['message_tree_id'])
        self.test_dataset = self.test_dataset.remove_columns(['message_tree_id'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class Model(pl.LightningModule):
    def __init__(self, model_name, batch_size, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.save_hyperparameters()

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        native_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
        self.model = get_peft_model(native_model, lora_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side='left')

        self.metric = evaluate.load('rouge')

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_nb):
        outputs = self(batch)

        self.log('train_loss', outputs.loss)

        return {'loss': outputs.loss}

    def validation_step(self, batch, batch_nb):
        outputs = self(batch)

        label_ids = batch['labels']
        # Replace -100 in the prediction with the pad token id in the tokenizer, otherwise an error occurs while
        # decoding
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        generated_ids = self.model.generate(**batch, max_new_tokens=200)
        label = self.tokenizer.batch_decode(label_ids)
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        rouge = self.metric.compute(predictions=generated_text, references=label)

        if batch_nb < 3:
            inputs = self.tokenizer.decode(batch['input_ids'][0])
            columns = ["Input", "Label", "Prediction"]
            data = [[inputs, label[0], generated_text[0]]]
            self.logger.log_text(key=f"Sample-Epoch{self.current_epoch}-Batch{batch_nb}", columns=columns, data=data)

        # Check accuracy and f1 score
        # final_score = self.metric.compute(predictions=generated_ids[0], references=label_ids)

        self.log('val_loss', outputs.loss)
        self.log('val_rouge1', rouge['rouge1'])
        self.log('val_rouge2', rouge['rouge2'])
        self.log('val_rougeL', rouge['rougeL'])
        return {'val_loss': outputs.loss, 'val_rouge1': rouge['rouge1'], 'val_rouge2': rouge['rouge2'], 'val_rougeL': rouge['rougeL']}

    def test_step(self, batch, batch_nb):
        outputs = self(batch)

        label_ids = batch['labels']
        # Replace -100 in the prediction with the pad token id in the tokenizer, otherwise an error occurs while
        # decoding
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        generated_ids = self.model.generate(**batch, max_new_tokens=200)
        label = self.tokenizer.batch_decode(label_ids)
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        rouge = self.metric.compute(predictions=generated_text, references=label)

        self.log('test_loss', outputs.loss)
        self.log('test_rouge1', rouge['rouge1'])
        self.log('test_rouge2', rouge['rouge2'])
        self.log('test_rougeL', rouge['rougeL'])
        return {'test_loss': outputs.loss, 'test_rouge1': rouge['rouge1'], 'test_rouge2': rouge['rouge2'], 'test_rougeL': rouge['rougeL']}

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)
