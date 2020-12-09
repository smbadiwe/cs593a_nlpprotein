# Some of the code adapted from https://github.com/alrojo/CB513/blob/master/data.py
import gzip
import h5py
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from os import path
import numpy as np
from util import print_info, AA_ID_DICT, ID_AA_DICT

# Details of the dataset at https://github.com/alrojo/CB513/

TRAIN_DOWNLOAD_LINK = "http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_5926_filtered.npy.gz"
TEST_DOWNLOAD_LINK = "http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz"

ROOT_DATA_PATH = "../data"
TRAIN_PATH = path.join(ROOT_DATA_PATH, "cullpdb+profile_5926_filtered.npy.gz")
TEST_PATH = path.join(ROOT_DATA_PATH, "cb513+profile_split1.npy.gz")

H5_FILE = path.join(ROOT_DATA_PATH, "cullpdb.hdf5")

N_AMINO_ACIDS = 700
N_FEATURES = 57


def get_train():
    ensure_data_exists(TRAIN_PATH, False)
    with gzip.open(TRAIN_PATH, "r") as f:
        X = np.load(f)
    X = X.reshape((-1, N_AMINO_ACIDS, N_FEATURES))
    X = X[:, :, :]
    labels = X[:, :, 22:30]
    mask = X[:, :, 30] * -1 + 1

    a = np.arange(0, 21)
    b = np.arange(35, 56)
    c = np.hstack((a, b))
    X = X[:, :, c]

    # getting meta
    num_seqs = np.size(X, 0)
    seqlen = np.size(X, 1)
    d = np.size(X, 2)
    num_classes = 8

    #### REMAKING LABELS ####
    X = X.astype(np.float)
    mask = mask.astype(np.float)
    # Dummy -> concat
    vals = np.arange(0, 8)
    labels_new = np.zeros((num_seqs, seqlen))
    for i in range(np.size(labels, axis=0)):
        labels_new[i, :] = np.dot(labels[i, :, :], vals)
    labels_new = labels_new.astype('int32')
    labels = labels_new

    print("Loading splits ...")
    ##### SPLITS #####

    X_train = X[:5278]
    X_valid = X[5278:]
    labels_train = labels[:5278]
    labels_valid = labels[5278:]
    mask_train = mask[:5278]
    mask_valid = mask[5278:]

    return X_train, X_valid, labels_train, labels_valid, mask_train, mask_valid


def get_test():
    ensure_data_exists(TEST_PATH, True)
    with gzip.open(TEST_PATH, "r") as f:
        X_test = np.load(f)
    X_test = X_test.reshape((-1, N_AMINO_ACIDS, N_FEATURES))
    X_test = X_test[:, :, :]
    labels_test = X_test[:, :, 22:30].astype('int32')
    mask_test = X_test[:, :, 30] * -1 + 1
    print(f"Check: {np.unique(X_test[0, 0, 35:])}")
    a = np.arange(0, 21)
    b = np.arange(35, 56)
    c = np.hstack((a, b))
    X_test = X_test[:, :, c]

    # getting meta
    seqlen = np.size(X_test, 1)
    d = np.size(X_test, 2)
    num_classes = 8
    num_seq_test = np.size(X_test, 0)
    del a, b, c

    ## DUMMY -> CONCAT ##
    vals = np.arange(0, 8)
    labels_new = np.zeros((num_seq_test, seqlen))
    for i in range(np.size(labels_test, axis=0)):
        labels_new[i, :] = np.dot(labels_test[i, :, :], vals)
    labels_new = labels_new.astype('int32')
    labels_test = labels_new

    ### ADDING BATCH PADDING ###

    X_add = np.zeros((126, seqlen, d))
    label_add = np.zeros((126, seqlen))
    mask_add = np.zeros((126, seqlen))

    X_test = np.concatenate((X_test, X_add), axis=0).astype(np.float)
    labels_test = np.concatenate((labels_test, label_add), axis=0).astype('int32')
    mask_test = np.concatenate((mask_test, mask_add), axis=0).astype(np.float)

    return X_test, mask_test, labels_test, num_seq_test


def ensure_data_exists(data_path, test):
    if path.exists(data_path):
        return

    import requests

    url = TEST_DOWNLOAD_LINK if test else TRAIN_DOWNLOAD_LINK
    print(f"Downloading {path.basename(data_path)} from {url}...")
    with requests.get(url, allow_redirects=True) as r:
        with gzip.open(data_path, 'w') as f:
            f.write(r.content)

    print(f"DONE Downloading {path.basename(data_path)} from {url}...")


def build_hdf5():
    if path.exists(H5_FILE):
        return

    X_train, X_valid, labels_train, labels_valid, mask_train, mask_valid = get_train()
    X_test, mask_test, labels_test, num_seq_test = get_test()
    unq = np.unique(X_test[:, :, 21:])
    print(f"# Uniques: {len(unq)} - ", unq)

    def create_data(ff, name, data):
        ff.create_dataset(name=name, data=data, compression='lzf', chunks=True)

    with h5py.File(H5_FILE, "w") as f:
        create_data(f, "/train/x", X_train)
        create_data(f, "/train/label", labels_train)
        create_data(f, "/train/mask", mask_train)
        create_data(f, "/val/x", X_valid)
        create_data(f, "/val/label", labels_valid)
        create_data(f, "/val/mask", mask_valid)
        create_data(f, "/test/x", X_test)
        create_data(f, "/test/label", labels_test)
        create_data(f, "/test/mask", mask_test)

        print_info(f)


class CullPDBDataset(Dataset):
    def __init__(self, data_split, tokenizer):
        assert data_split in ['train', 'val', 'test']
        self.tokenizer = tokenizer
        if tokenizer:
            PSSMs = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y',
                     'X']
            self.PSSMs = {i: s for i, s in enumerate(PSSMs)}
            self.SEQ = {i: s for i, s in enumerate("ACDEFGHIKLMNPQRSTVWXY")}

        self.data_split = data_split
        build_hdf5()
        with h5py.File(H5_FILE, "r") as f:
            self.length = f[f"/{data_split}/label"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        ds = self.data_split
        with h5py.File(H5_FILE, "r") as f:
            X = f[f"/{ds}/x"][item, :, :]
            label = f[f"/{ds}/label"][item]
            mask = f[f"/{ds}/mask"][item]
        if True or self.tokenizer:
            # convert them back to amino acids, ignoring the embedding in the data we have
            # pssm_idx = np.arange(0, 21)
            print("Uniques:", np.unique(X[:, 21:]))
            seq_idx = np.argmax(X[:, 21:], axis=1)
            assert seq_idx.shape == (X.shape[0],), f"Idx shape: {seq_idx.shape}"
            # pssm_str = " ".join([self.PSSMs[i] for i in pssm_idx])

        return {
            "X": X,
            "label": label,
            "mask": mask
        }


if __name__ == "__main__":
    # print("# PSSMs:", len(PSSMs))
    # print("# SEQ:", len(SEQ))
    # with h5py.File(H5_FILE, "r") as f:
    #     x = f[f"/test/label"][:]
    #     print(f"label: ({x.shape}) {np.unique(x)}")
    #     x = f[f"/test/mask"][:]
    #     print(f"mask: ({x.shape}) {np.unique(x)}")
    #     x = f[f"/test/x"][:]
    #     print(f"x: ({x.shape}) {np.unique(x)}")
    build_hdf5()
    # ds = CullPDBDataset("test", None)
    # val = ds[9]
    # for k, v in val.items():
    #     print(f"{k}: {v.shape}")
