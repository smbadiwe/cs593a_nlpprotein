# See https://huggingface.co/Rostlab/prot_bert
from os import path, listdir, makedirs
import numpy as np
# import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertTokenizer, BertConfig
import re

PRE_TRAINED_MODEL_NAME = "Rostlab/prot_bert"

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=False)
model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
sequence_Example = "A E T C Z A O"
sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
encoded_input = tokenizer(sequence_Example, return_tensors='pt')
output = model(**encoded_input)

print(encoded_input)


# print(output[0].shape)
# print(output[1].shape)


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