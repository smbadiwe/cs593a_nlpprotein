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
    runner = RunnerForNetSurf2(**expt_prot_bert_bfd)
    runner.train()


def test():
    runner = RunnerForNetSurf2(**expt_prot_bert_bfd)
    test_files = ['CB513_HHblits.csv']
    for test_file in test_files:
        print(f'\n{test_file} RESULTS:')
        d_set = runner.load_dataset(test_file)
        runner.test(d_set)


if __name__ == '__main__':
    test()
