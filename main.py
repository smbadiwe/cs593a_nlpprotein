from prottrans_ss import RunnerForNetSurf2
from set2018 import RunnerForSet2018

expt_prot_bert = {
    'experiment_name': "prot_bert",
    'n_labels': 8,
    'model_name': "Rostlab/prot_bert",
    'max_length': 512  # 1024
}

expt_prot_bert_bfd = {
    'experiment_name': "prot_bert_bfd",
    'n_labels': 8,
    'model_name': "Rostlab/prot_bert_bfd",
    'max_length': 512  # 1024
}

expt_prot_albert = {
    'experiment_name': "prot_albert",
    'n_labels': 8,
    'model_name': "Rostlab/prot_albert",
    'max_length': 1024
}

"""
Error running alpert
Exception: You're trying to run a `Unigram` model but you're file was trained with a different algorithm
"""


def train():
    runner = RunnerForNetSurf2(**expt_prot_albert)
    runner.train()


if __name__ == '__main__':
    train()
