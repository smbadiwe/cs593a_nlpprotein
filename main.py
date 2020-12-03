from transformers import BertModel, BertTokenizer, BertConfig
from protein_dataset import ProteinNetDataset
PRE_TRAINED_MODEL_NAME = "Rostlab/prot_bert"

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=False )


if __name__ == '__main__':

    dd = ProteinNetDataset("training_70", tokenizer)
    print(dd[5]["sequence"])
    print(dd[5]["encoding"])
