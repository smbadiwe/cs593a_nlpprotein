"""
Paper: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0232528
Github: https://github.com/sh-maxim/ss

"""

from transformers import Trainer
from ssp_dataset import SSPDataset
from util import datasetFolderPath
from os import path
import re
import pandas as pd
from huggingface_runner import HuggingFaceRunner

# Details of the dataset at https://github.com/alrojo/CB513/

DOWNLOAD_LINK = "https://raw.githubusercontent.com/sh-maxim/ss/master/S2_Data_set.txt"
TXT_FILE = path.join(datasetFolderPath, "set2018.txt")


def ensure_data_exists():
    if path.exists(TXT_FILE):
        return

    import requests

    print(f"Downloading {path.basename(TXT_FILE)} from {DOWNLOAD_LINK}...")
    with requests.get(DOWNLOAD_LINK, allow_redirects=True) as r:
        with open(TXT_FILE, 'wb') as f:
            f.write(r.content)

    print(f"DONE Downloading {path.basename(TXT_FILE)} from {DOWNLOAD_LINK}...")


def build_hdf5():
    ln = 0
    train_seq, val_seq, test_seq = [], [], []
    train_label_3, val_label_3, test_label_3 = [], [], []
    train_label_8, val_label_8, test_label_8 = [], [], []

    # label_3 is Rule_1_3_label_ss

    with open(TXT_FILE, 'r') as f:
        for line in f:
            ln += 1
            if not line or line.startswith("#"):
                continue

            spl = line.split("\t")
            # assert len(spl) == 10, f"Line {ln}: # cols: {len(spl)}\n{line}"

            data_div = spl[0]
            # assert data_div in ["test", "valid", "train"], f"Data_div is {data_div}"
            # assert len(spl[4]) == len(spl[5]) == len(spl[6])
            if data_div == 'train':
                train_seq.append(spl[4])
                train_label_8.append(spl[5])
                train_label_3.append(spl[6])

            elif data_div == 'val':
                val_seq.append(spl[4])
                val_label_8.append(spl[5])
                val_label_3.append(spl[6])

            elif data_div == 'test':
                test_seq.append(spl[4])
                test_label_8.append(spl[5])
                test_label_3.append(spl[6])

    return {
        "train": (train_seq, train_label_3, train_label_8),
        "val": (val_seq, val_label_3, val_label_8),
        "test": (test_seq, test_label_3, test_label_8),
    }


class Set2018DatasetLoader:
    def __init__(self, max_length, n_labels=8):
        self.n_labels = n_labels
        self.max_length = max_length
        self.df: 'pd.DataFrame' = None
        self.data_split = "train"

    def _load(self, file_path):
        dssp = "DSSP_8_label_ss" if self.n_labels == 8 else "Rule_1_3_label_ss"
        df = pd.read_csv(file_path, comment="#", sep='\t',
                         usecols=['subset_type', 'complete_aa_seq', dssp])
        # print(df.info())
        # print()
        # return None
        print(f"{file_path} dataset columns:\n", df.columns.tolist())

        df['input_fixed'] = ["".join(seq.split()) for seq in df['complete_aa_seq']]
        df['input_fixed'] = [re.sub(r"[UZOB]", "X", seq) for seq in df['input_fixed']]

        df['label_fixed'] = ["".join(label.split()) for label in df[dssp]]
        self.df = df

    def load_dataset(self, file_path) -> tuple:
        if self.df is None:
            self._load(file_path)
        print(f"Loading {self.data_split} data...")

        df = self.df[self.df['subset_type'] == self.data_split]
        seqs = [list(seq)[:self.max_length - 2] for seq in df['input_fixed']]

        labels = [list(label)[:self.max_length - 2] for label in df['label_fixed']]

        disorder = [(['1.0'] * min(len(disorder), self.max_length - 2)) for disorder in seqs]

        assert len(seqs) == len(labels) == len(disorder)
        return seqs, labels, disorder


class RunnerForSet2018(HuggingFaceRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_dir = self.results_dir.replace('./results', './results-set2018')
        self.logs_dir = self.results_dir.replace('./logs', './logs-set2018')
        ensure_data_exists()

    @property
    def dataset_loader(self):
        if self._loader is None:
            self._loader = Set2018DatasetLoader(max_length=self.max_length, n_labels=self.n_labels)

        return self._loader

    def _get_dataset(self, key: str) -> 'SSPDataset':
        assert key in ['train', 'valid', 'test'], f"Value for key should be train, valid or test; not {key}."
        if self._loader is None:
            _ = self.dataset_loader
        self._loader.data_split = key
        return self.load_dataset("set2018.txt")

    def train(self, model=None) -> 'Trainer':
        train_data = self._get_dataset("train")
        val_data = self._get_dataset("valid")
        return self.do_training(train_data=train_data, val_data=val_data, model=model)


if __name__ == "__main__":
    runner = Set2018DatasetLoader(max_length=1024, n_labels=8)
    for opt in ['train', 'valid', 'test']:
        runner.data_split = opt
        sss, lll, ddd = runner.load_dataset(TXT_FILE)
        print(f"# {runner.data_split}: {len(sss):,}")
        for s, l, d in zip(sss, lll, ddd):
            assert len(s) == len(l) == len(d)
