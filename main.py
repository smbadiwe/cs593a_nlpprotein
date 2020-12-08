from netsurfp2 import RunnerForNetSurfp2, NetSurfp2DatasetLoader
from set2018 import RunnerForSet2018
import argparse

"""

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

expt_prot_albert = {  # does not work
    'experiment_name': "prot_albert",
    'n_labels': 8,
    'model_name': "Rostlab/prot_albert",
    'max_length': 1024
}

Error running alpert
Exception: You're trying to run a `Unigram` model but you're file was trained with a different algorithm
"""


def get_args() -> tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--maxlen', type=int, default=512,
                        help='Max sequence length. 1024 is used in the model but may not run on low-resource machine.')
    parser.add_argument('-l', '--labels', type=int, default=8,
                        help="Number of labels. Should be 3 or 8. Model will add 1 to this count to cater for unknowns.")
    parser.add_argument('-e', '--experiment', type=str, default='prot_bert', help="Name of experiment")
    parser.add_argument('-b', '--bfd', action='store_true', default=False,
                        help="If set, use the BFD version of the prot_bert model. Otherwise, use the UniRef100 version")
    parser.add_argument('-s', '--set2018', action='store_true', default=False)
    parser.add_argument('-t', '--test', action='store_true', default=False,
                        help="If set, test model on available datasets. Otherwise, train.")
    args = parser.parse_args()
    data = {
        'model_name': "Rostlab/prot_bert",
        'experiment_name': args.experiment,
        'n_labels': args.labels,
        'max_length': args.maxlen  # 512
    }
    if args.bfd:
        data['model_name'] = data['model_name'] + '_bfd'
        data['experiment_name'] = data['experiment_name'] + '_bfd'
    if args.set2018:
        klass = RunnerForSet2018
    else:
        klass = RunnerForNetSurfp2
    print("args:", args)
    return klass, data, args.test


def train(Class, data):
    runner = Class(**data)
    runner.train()


def test(Class, data):
    results = {}
    runner = Class(**data)

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
    klass_, data_, testing = get_args()
    if testing:
        test(klass_, data_)
    else:
        train(klass_, data_)

    # prot_bert 512 netsurf done
