import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForTokenClassification, BertTokenizerFast, \
    EvalPrediction
from torch.utils.data import Dataset
import os
from os import path
import pandas as pd
import requests
from tqdm.auto import tqdm
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import re
from util import DATASETS_AND_PATHS, AA_ID_DICT, ID_AA_DICT

datasetFolderPath = "dataset/"


def download_data(key):
    if not os.path.exists(datasetFolderPath):
        os.makedirs(datasetFolderPath)

    def download_file(url, filename):
        print(f"Downloading file {filename} from {url}...")
        response = requests.get(url, stream=True)
        with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                           total=int(response.headers.get('content-length', 0)),
                           desc=filename) as fout:
            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)

    url_path, file = DATASETS_AND_PATHS[key]
    file = os.path.join(datasetFolderPath, file)
    if not os.path.exists(file):
        if url_path:
            download_file(url_path, file)
        else:  # combine all test dataset files. To be used as val. pd.read_csv
            print("combine all test dataset files. To be used as val.")
            concats = []
            for k, f in DATASETS_AND_PATHS.items():
                if f[0] and k.endswith('test'):
                    the_file = os.path.join(datasetFolderPath, f[1])
                    print("COMBINE: Loading file ", the_file)
                    concats.append(pd.read_csv(the_file))
            combined_csv = pd.concat(concats)
            # combined_csv = pd.concat(
            #     [pd.read_csv(os.path.join(datasetFolderPath, f[1])) for k, f in DATASETS_AND_PATHS.items() if f[0] and k.endswith('test')])
            # export to csv
            combined_csv.to_csv(file, index=False, encoding='utf-8-sig')


def download_netsurfp_dataset():
    for k in DATASETS_AND_PATHS:
        download_data(k)


def mask_disorder(labels, masks):
    for label, mask in zip(labels, masks):
        for i, disorder in enumerate(mask):
            if disorder == "0.0":
                # shift by one because of the CLS token at index 0
                label[i + 1] = -100


def encode_tags(tags, encodings):
    labels = [[AA_ID_DICT[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


def align_predictions(predictions: np.ndarray, label_ids: np.ndarray):
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(ID_AA_DICT[label_ids[i][j]])
                preds_list[i].append(ID_AA_DICT[preds[i][j]])

    return preds_list, out_label_list


def compute_metrics(p: EvalPrediction):
    preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
    return {
        "accuracy": accuracy_score(out_label_list, preds_list),
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }


class SS3Dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class HuggingFaceRunner:
    def __init__(self, experiment_name, n_labels=3, model_name="Rostlab/prot_bert", max_length=1024):
        assert n_labels in [3, 8], f"n_labels should be 3 or 8, not {n_labels}"
        self.model_name = model_name
        self.n_labels = n_labels
        if experiment_name is None:
            experiment_name = model_name.split('/')[-1]
        self.experiment_name = experiment_name
        self.max_length = max_length
        self.results_dir = path.join('./results', model_name, f"SS{n_labels}-{max_length}", experiment_name)
        self.logs_dir = path.join('./logs', model_name, f"SS{n_labels}-{max_length}", experiment_name)
        try:
            self.seq_tokenizer = BertTokenizerFast.from_pretrained(self.results_dir,
                                                                   do_lower_case=False)
        except Exception as e:
            print(f"Failure loading tokenizer from {self.results_dir}. It probably doesn't exist yet.")
            print(e, f"Loading tokenizer from {model_name}...")
            self.seq_tokenizer = BertTokenizerFast.from_pretrained(model_name,
                                                                   do_lower_case=False)
        self.id2tag: dict = None
        self.tag2id: dict = None
        download_netsurfp_dataset()

    def get_trainer(self, model_init, train_dataset, val_dataset, experiment_name=None) -> 'Trainer':
        if not experiment_name:
            experiment_name = self.model_name.split('/')[-1]

        training_args = TrainingArguments(
            output_dir=self.results_dir,  # output directory
            num_train_epochs=3,  # total number of training epochs
            per_device_train_batch_size=1,  # batch size per device during training
            per_device_eval_batch_size=8,  # batch size for evaluation
            warmup_steps=200,  # number of warmup steps for learning rate scheduler
            learning_rate=3e-05,  # learning rate
            weight_decay=0.0,  # strength of weight decay
            logging_dir=self.logs_dir,  # directory for storing logs
            logging_steps=200,  # How often to print logs
            do_train=True,  # Perform training
            do_eval=(val_dataset is not None),  # Perform evaluation
            evaluation_strategy="epoch",  # evaluate after each epoch
            gradient_accumulation_steps=32,  # total number of steps before back propagation
            fp16=True,  # Use mixed precision
            fp16_opt_level="02",  # mixed precision mode
            run_name=experiment_name,  # experiment name
            seed=3,  # Seed for experiment reproducibility
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
        )

        trainer = Trainer(
            model_init=model_init,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,  # evaluation metrics
        )

        return trainer

    def load_dataset(self, file_path) -> tuple:
        dssp = f'dssp{self.n_labels}'
        df = pd.read_csv(file_path, skiprows=1,
                         names=['input', dssp, 'disorder', 'cb513_mask'])
        print(f"{file_path} dataset columns:\n", df.columns.tolist())
        df['input_fixed'] = ["".join(seq.split()) for seq in df['input']]
        df['input_fixed'] = [re.sub(r"[UZOB]", "X", seq) for seq in df['input_fixed']]
        seqs = [list(seq)[:self.max_length - 2] for seq in df['input_fixed']]

        df['label_fixed'] = ["".join(label.split()) for label in df[dssp]]
        labels = [list(label)[:self.max_length - 2] for label in df['label_fixed']]

        df['disorder_fixed'] = [" ".join(disorder.split()) for disorder in df['disorder']]
        disorder = [disorder.split()[:self.max_length - 2] for disorder in df['disorder_fixed']]

        assert len(seqs) == len(labels) == len(disorder)

        return seqs, labels, disorder

    def get_dataset(self, key: str) -> 'SS3Dataset':
        _, file = DATASETS_AND_PATHS[key]
        seqs, labels, disorder = self.load_dataset(path.join(datasetFolderPath, file))

        unique_tags = set(tag for doc in labels for tag in doc)
        print(f"Key: {key}. Unique Tags: {len(unique_tags)}\n", unique_tags)
        if self.tag2id is None and key.endswith('test'):
            # Consider each label as a tag for each token
            self.n_labels = len(unique_tags)
            self.tag2id = {tag: i for i, tag in enumerate(unique_tags)}
            self.id2tag = {i: tag for tag, i in self.tag2id.items()}

        seqs_encodings = self.seq_tokenizer(seqs, is_split_into_words=True,
                                            return_offsets_mapping=True,
                                            truncation=True, padding=True)

        labels_encodings = encode_tags(labels, seqs_encodings)
        mask_disorder(labels_encodings, disorder)
        _ = seqs_encodings.pop("offset_mapping")

        return SS3Dataset(seqs_encodings, labels_encodings)

    def train(self, model=None) -> 'Trainer':
        train_data = self.get_dataset("netsurfp")
        val_data = self.get_dataset("combinedtest")

        if model is not None:
            def model():
                try:
                    return AutoModelForTokenClassification.from_pretrained(self.results_dir,
                                                                           num_labels=self.n_labels,
                                                                           id2label=self.id2tag,
                                                                           label2id=self.tag2id,
                                                                           gradient_checkpointing=False)

                except Exception as e:
                    print(f"Failure loading model from {self.results_dir}. It probably doesn't exist yet.")
                    print(e, f"Loading model from {self.model_name}...")

                    return AutoModelForTokenClassification.from_pretrained(self.model_name,
                                                                           num_labels=self.n_labels,
                                                                           id2label=self.id2tag,
                                                                           label2id=self.tag2id,
                                                                           gradient_checkpointing=False)

        trainer = self.get_trainer(model, train_dataset=train_data,
                                   val_dataset=val_data)
        trainer.train()

        trainer.save_model(self.results_dir)
        fs = self.seq_tokenizer.save_pretrained(self.results_dir)
        print(f"Saved model and tokenizer. Tokenizer files saved:\n", fs)
        return trainer

    def test(self, trainer, dataset_key="casp12test"):
        test_dataset = self.get_dataset(dataset_key)
        predictions, label_ids, metrics = trainer.predict(test_dataset)


if __name__ == "__main__":
    pass