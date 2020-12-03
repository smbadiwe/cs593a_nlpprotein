import h5py
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from os import path
from util import BasicSequenceTokenizer, MAX_SEQUENCE_LENGTH, DATA_PATH


def print_info(h5_obj, file_sep='\t'):
    def __print_details(obj, sep):
        """
        Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
        """
        if type(obj) in [h5py.Group, h5py.File]:
            for key in obj.keys():
                print(sep, '-', key, ':', obj[key])
                __print_details(obj[key], sep=sep + '\t')
        elif type(obj) == h5py.Dataset:
            for key in obj.attrs.keys():
                print(sep + '\t', '-', key, ':', obj.attrs[key])

    __print_details(h5_obj['/'], file_sep)


class ProteinNetDataset(Dataset):
    def __init__(self, hfd5_file, tokenizer):
        self.tokenizer = tokenizer
        self.file_path = path.join(DATA_PATH, hfd5_file + ".hdf5")

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
