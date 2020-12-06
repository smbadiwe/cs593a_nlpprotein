from prottrans_ss import HuggingFaceRunner

expt_1 = {
    'experiment_name': "prot_bert",
    'n_labels': 8,
    'model_name': "Rostlab/prot_bert",
    'max_length': 512  # 1024
}

expt_2 = {
    'experiment_name': "prot_bert_bfd",
    'n_labels': 8,
    'model_name': "Rostlab/prot_bert_bfd",
    'max_length': 512  # 1024
}


def run():
    runner = HuggingFaceRunner(**expt_2)
    runner.train()


if __name__ == '__main__':
    run()
