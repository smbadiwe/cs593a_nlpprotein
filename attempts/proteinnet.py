import h5py
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from os import path
from util import BasicSequenceTokenizer


class ProteinNetDataset(Dataset):
    MAX_SEQUENCE_LENGTH = 2000
    DATA_PATH = "../cs593a_proteins/data/preprocessed/casp11"

    def __init__(self, hfd5_file, tokenizer):
        self.tokenizer = tokenizer
        self.file_path = path.join(self.DATA_PATH, hfd5_file + ".hdf5")

        with h5py.File(self.file_path, "r") as f:
            self.length = f["/primary"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        with h5py.File(self.file_path, "r") as f:
            sequence = BasicSequenceTokenizer.decode(f["/primary"][item, :], sep=" ")

        encoding = self.tokenizer(sequence, return_tensors='pt')

        return {
            'sequence': sequence,
            'encoding': encoding
        }


def create_data_loader(hfd5_file, tokenizer, batch_size):
    ds = ProteinNetDataset(hfd5_file=hfd5_file, tokenizer=tokenizer)

    return DataLoader(ds, batch_size=batch_size, num_workers=4)


if __name__ == "__main__":
    pass
