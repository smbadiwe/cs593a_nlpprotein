# See https://huggingface.co/Rostlab/prot_bert
from os import path, listdir, makedirs
from torch.utils.data import Dataset
import numpy as np
from prottrans_ss import HuggingFaceRunner
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig, Trainer, TrainingArguments
import re
from typing import Optional, Dict, List, Tuple, Union, Any


class ThisTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, model_path: Optional[str] = None, trial: Dict[str, Any] = None):
        print(f"ThisTrainer: Training now...")
        super().train(model_path=model_path, trial=trial)

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        print(f"ThisTrainer: Evaluating now...")
        return super().evaluate(eval_dataset=eval_dataset)


class PureRunner(HuggingFaceRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_trainer(self, model_init, train_dataset, val_dataset, experiment_name=None) -> 'Trainer':
        if not experiment_name:
            experiment_name = self.model_name.split('/')[-1]

        training_args = TrainingArguments(
            output_dir=self.results_dir,  # output directory
            num_train_epochs=3,  # total number of training epochs
            per_device_train_batch_size=1,  # batch size per device during training
            per_device_eval_batch_size=8,  # batch size for evaluation
            warmup_steps=200,  # number of warmup steps for learning rate scheduler
            learning_rate=3e-05,  # learning rate
            weight_decay=0.0,  # strength of weight decay
            logging_dir=self.logs_dir,  # directory for storing logs
            logging_steps=200,  # How often to print logs
            do_train=True,  # Perform training
            do_eval=(val_dataset is not None),  # Perform evaluation
            evaluation_strategy="epoch",  # evaluate after each epoch
            gradient_accumulation_steps=32,  # total number of steps before back propagation
            fp16=False,  # True,  # Use mixed precision
            fp16_opt_level="02",  # mixed precision mode
            run_name=experiment_name,  # experiment name
            seed=3,  # Seed for experiment reproducibility
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
        )

        trainer = ThisTrainer(
            model_init=model_init,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,  # evaluation metrics
        )

        return trainer




PRE_TRAINED_MODEL_NAME = "Rostlab/prot_bert"

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=False)
model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
sequence_Example = "A E T C Z A O"
sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
encoded_input = tokenizer(sequence_Example, return_tensors='pt')
output = model(**encoded_input)

print(encoded_input)


class GenresClassifier(nn.Module):

    def __init__(self, n_classes):
        super(GenresClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        output = self.drop(pooled_output)
        return self.out(output)


def train(epoch):
    losses = []

    model.train()
    for data in train_data_loader:
        ids = data['input_ids'].to(device)
        mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        targets = data['genres'].to(device)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_fn(outputs, targets)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(losses)


def validation(epoch):
    losses = []

    model.eval()
    with torch.no_grad():
        for data in val_data_loader:
            ids = data['input_ids'].to(device)
            mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            targets = data['genres'].to(device)

            outputs = model(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)
            losses.append(loss.item())

    return np.mean(losses)