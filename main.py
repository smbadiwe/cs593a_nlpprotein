from prottrans_ss import HuggingFaceRunner
model_name = "Rostlab/prot_bert"  # 'Rostlab/prot_bert_bfd'

max_length = 1024

expt_1 = {
    'experiment_name': "prot_bert",
    'n_labels': 8,
    'model_name': "Rostlab/prot_bert",
    'max_length': 512  # 1024
}


def run():
    runner = HuggingFaceRunner(**expt_1)
    runner.train()


if __name__ == '__main__':
    run()
