import h5py

AA_ID_DICT = {'': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
              'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17,
              'V': 18, 'W': 19, 'Y': 20}
ID_AA_DICT = {0: '', 1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
              10: 'L', 11: 'M', 12: 'N', 13: 'P', 14: 'Q', 15: 'R', 16: 'S', 17: 'T',
              18: 'V', 19: 'W', 20: 'Y'}


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


class BasicSequenceTokenizer:
    @staticmethod
    def encode(sequence: str) -> list:
        return list([AA_ID_DICT[aa] for aa in sequence])

    @staticmethod
    def decode(encoded_sequence: list, sep=" ") -> str:
        return sep.join([ID_AA_DICT[aa] for aa in encoded_sequence])

# rev = {v:k for k, v in AA_ID_DICT.items()}
# print(rev)


DATASETS_AND_PATHS = {
    'netsurfp': ('https://www.dropbox.com/s/98hovta9qjmmiby/Train_HHblits.csv?dl=1',
                 'Train_HHblits.csv'),
    'ts115test': ('https://www.dropbox.com/s/68pknljl9la8ax3/TS115_HHblits.csv?dl=1',
                  'TS115_HHblits.csv'),
    'cb513test': ('https://www.dropbox.com/s/9mat2fqqkcvdr67/CB513_HHblits.csv?dl=1',
                  'CB513_HHblits.csv'),
    'casp12test': ('https://www.dropbox.com/s/te0vn0t7ocdkra7/CASP12_HHblits.csv?dl=1',
                   'CASP12_HHblits.csv'),
    'combinedtest': ('', 'Validation_HHblits.csv')
}

datasetFolderPath = "dataset/"
