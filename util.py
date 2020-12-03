
MAX_SEQUENCE_LENGTH = 2000
DATA_PATH = "../cs593a_proteins/data/preprocessed/casp11"

AA_ID_DICT = {'': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
              'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17,
              'V': 18, 'W': 19, 'Y': 20}
ID_AA_DICT = {0: '', 1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
              10: 'L', 11: 'M', 12: 'N', 13: 'P', 14: 'Q', 15: 'R', 16: 'S', 17: 'T',
              18: 'V', 19: 'W', 20: 'Y'}


class BasicSequenceTokenizer:
    @staticmethod
    def encode(sequence: str) -> list:
        return list([AA_ID_DICT[aa] for aa in sequence])

    @staticmethod
    def decode(encoded_sequence: list, sep=" ") -> str:
        return sep.join([ID_AA_DICT[aa] for aa in encoded_sequence])

# rev = {v:k for k, v in AA_ID_DICT.items()}
# print(rev)
