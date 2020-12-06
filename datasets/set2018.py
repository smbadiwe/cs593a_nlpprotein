"""
Paper: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0232528
Github: https://github.com/sh-maxim/ss

"""

import gzip
import h5py
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from os import path
import numpy as np
from util import print_info, AA_ID_DICT, ID_AA_DICT

# Details of the dataset at https://github.com/alrojo/CB513/

DOWNLOAD_LINK = "https://raw.githubusercontent.com/sh-maxim/ss/master/S2_Data_set.txt"
ROOT_DATA_PATH = "../data"

TXT_FILE = path.join(ROOT_DATA_PATH, "set2018.txt")
H5_FILE = path.join(ROOT_DATA_PATH, "set2018.hdf5")

LABEL_3_MAP = {'X': 0, 'H': 1, 'E': 2, 'C': 3}
LABEL_8_MAP = {'X': 0, 'H': 1, 'E': 2, 'C': 3, 'T': 4, 'G': 5, 'S': 6, 'B': 7, 'I': 8}


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
    if path.exists(H5_FILE):
        return

    ensure_data_exists()

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


class Sep2018Dataset(Dataset):
    def __init__(self, sequences, labels, tokenizer):
        self.tokenizer = tokenizer
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        sequence = " ".join(self.sequences[item])
        label = self.labels[item]

        return {
            "X": X,
            "label": label,
            "mask": mask
        }


if __name__ == "__main__":
    build_hdf5()
