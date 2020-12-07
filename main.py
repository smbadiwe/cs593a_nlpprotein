from prottrans_ss import RunnerForNetSurf2, NetSurf2DatasetLoader
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

"""

expt_prot_albert = {
    'experiment_name': "prot_albert",
    'n_labels': 8,
    'model_name': "Rostlab/prot_albert",
    'max_length': 1024
}

Error running alpert
Exception: You're trying to run a `Unigram` model but you're file was trained with a different algorithm
"""


def train():
    runner = RunnerForNetSurf2(**expt_prot_bert_bfd)
    runner.train()


def test():
    results = {}
    runner = RunnerForNetSurf2(**expt_prot_bert_bfd)

    n_params = sum(p.numel() for p in runner.model().parameters())
    print(f"# params: {n_params:,}")

    test_files = ['CB513_HHblits.csv', 'CASP12_HHblits.csv', 'TS115_HHblits.csv']
    for test_file in test_files:
        d_set = runner.load_dataset(test_file)
        metrics = runner.test(d_set)
        results[test_file] = metrics

    for k, v in results.items():
        print(f"{k}:\n{v}")


if __name__ == '__main__':
    train()
