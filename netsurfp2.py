"""
Some intuition from
https://github.com/agemagician/ProtTrans/blob/master/Fine-Tuning/ProtBert-BFD-FineTune-SS3.ipynb
"""
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForTokenClassification, BertTokenizerFast, \
    EvalPrediction
from torch.utils.data import Dataset
import os
from os import path
import pandas as pd
import requests
from tqdm.auto import tqdm
import numpy as np
import re
from util import DATASETS_AND_PATHS, datasetFolderPath, mask_disorder
from huggingface_runner import HuggingFaceRunner
from ssp_dataset import SSPDataset


def download_data(key):
    if not os.path.exists(datasetFolderPath):
        os.makedirs(datasetFolderPath)

    def download_file(url, filename):
        print(f"Downloading file {filename} from {url}...")
        response = requests.get(url, stream=True)
        with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                           total=int(response.headers.get('content-length', 0)),
                           desc=filename) as fout:
            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)

    url_path, file = DATASETS_AND_PATHS[key]
    file = os.path.join(datasetFolderPath, file)
    if not os.path.exists(file):
        if url_path:
            download_file(url_path, file)
        else:  # combine all test dataset files. To be used as val. pd.read_csv
            print("combine all test dataset files. To be used as val.")
            concats = []
            for k, f in DATASETS_AND_PATHS.items():
                if f[0] and k.endswith('test'):
                    the_file = os.path.join(datasetFolderPath, f[1])
                    print("COMBINE: Loading file ", the_file)
                    concats.append(pd.read_csv(the_file))
            combined_csv = pd.concat(concats)
            # combined_csv = pd.concat(
            #     [pd.read_csv(os.path.join(datasetFolderPath, f[1])) for k, f in DATASETS_AND_PATHS.items() if f[0] and k.endswith('test')])
            # export to csv
            combined_csv.to_csv(file, index=False, encoding='utf-8-sig')


def download_netsurfp_dataset():
    for k in DATASETS_AND_PATHS:
        download_data(k)


class NetSurfp2DatasetLoader:
    def __init__(self, max_length, n_labels=8):
        self.n_labels = n_labels
        self.max_length = max_length

    def load_dataset(self, file_path) -> tuple:
        dssp = f'dssp{self.n_labels}'
        df = pd.read_csv(file_path, skiprows=1, names=['input', dssp, 'disorder', 'cb513_mask'])
        print(f"{file_path} dataset columns:\n", df.columns.tolist())

        df['input_fixed'] = ["".join(seq.split()) for seq in df['input']]
        df['input_fixed'] = [re.sub(r"[UZOB]", "X", seq) for seq in df['input_fixed']]
        seqs = [list(seq)[:self.max_length - 2] for seq in df['input_fixed']]

        df['label_fixed'] = ["".join(label.split()) for label in df[dssp]]
        labels = [list(label)[:self.max_length - 2] for label in df['label_fixed']]

        df['disorder_fixed'] = [" ".join(disorder.split()) for disorder in df['disorder']]
        disorder = [disorder.split()[:self.max_length - 2] for disorder in df['disorder_fixed']]

        assert len(seqs) == len(labels) == len(disorder)

        return seqs, labels, disorder


class RunnerForNetSurfp2(HuggingFaceRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_dir = self.results_dir.replace('./results', './results-netsurf2')
        self.logs_dir = self.results_dir.replace('./logs', './logs-netsurf2')
        download_netsurfp_dataset()

    @property
    def dataset_loader(self):
        return NetSurfp2DatasetLoader(max_length=self.max_length, n_labels=self.n_labels)

    def _get_dataset(self, key: str) -> 'SSPDataset':
        _, file = DATASETS_AND_PATHS[key]
        return self.load_dataset(file)

    def train(self, model=None) -> 'Trainer':
        train_data = self._get_dataset("netsurfp")
        val_data = self._get_dataset("combinedtest")
        return self.do_training(train_data=train_data, val_data=val_data, model=model)


if __name__ == "__main__":
    df = pd.read_csv(path.join(datasetFolderPath, 'Train_HHblits.csv'))
    print(df.info())

