"""
Some intuition from
https://github.com/agemagician/ProtTrans/blob/master/Fine-Tuning/ProtBert-BFD-FineTune-SS3.ipynb
"""
from abc import ABC, abstractmethod
from os import path

import numpy as np
import torch
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import Trainer, TrainingArguments, AutoModelForTokenClassification, EvalPrediction, AutoTokenizer
from ssp_dataset import SSPDataset
from util import datasetFolderPath, mask_disorder


class HuggingFaceRunner(ABC):
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
        self._tokenizer: 'AutoTokenizer' = None
        self._model = None
        self._loader = None
        # X added for unknowns
        if n_labels == 8:
            unique_tags = ['X', 'B', 'C', 'E', 'G', 'H', 'I', 'S', 'T']
        else:
            unique_tags = ['X', 'C', 'E', 'H"']

        self.tag2id = {tag: i for i, tag in enumerate(unique_tags)}
        self.id2tag = {i: tag for tag, i in self.tag2id.items()}

        self.eval_mode = False

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.results_dir, do_lower_case=False, use_fast=True)
            except Exception as e:
                print(f"Failure loading tokenizer from {self.results_dir}. It probably doesn't exist yet.")
                print(e, f"Loading tokenizer from {self.model_name}...")
                if self.eval_mode:
                    raise e
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False, use_fast=True)

            print("TOKENIZER: ", self._tokenizer.__class__.__name__)
        return self._tokenizer

    def encode_tags(self, tags, encodings) -> list:
        labels = [[self.tag2id[tag] for tag in doc] for doc in tags]
        encoded_labels = []
        for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
            # create an empty array of -100
            doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
            arr_offset = np.array(doc_offset)

            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())

        return encoded_labels

    def get_trainer(self, model_init, train_dataset, val_dataset) -> 'Trainer':
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
            fp16=False,  # True,  # Use mixed precision
            fp16_opt_level="02",  # mixed precision mode
            run_name=self.experiment_name,  # experiment name
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
            compute_metrics=self.compute_metrics,  # evaluation metrics
        )

        return trainer

    @property
    def dataset_loader(self):
        if self._loader is not None:
            return self._loader
        raise NotImplementedError("dataset_loader not implemented")

    @abstractmethod
    def _get_dataset(self, key: str) -> 'SSPDataset':
        raise NotImplementedError("_get_dataset() not implemented")

    @abstractmethod
    def train(self, model=None) -> 'Trainer':
        raise NotImplementedError("train() not implemented")

    def test(self, test_dataset) -> dict:
        self.eval_mode = True
        if isinstance(test_dataset, str):
            test_dataset = self._get_dataset(test_dataset)
        trainer = self.get_trainer(model_init=self.model, train_dataset=None, val_dataset=None)
        predictions, label_ids, metrics = trainer.predict(test_dataset)
        self.eval_mode = False
        return metrics

    def load_dataset(self, file) -> 'SSPDataset':
        seqs, labels, disorder = self.dataset_loader.load_dataset(path.join(datasetFolderPath, file))

        if self.tag2id is None:
            # Consider each label as a tag for each token
            unique_tags = set(tag for doc in labels for tag in doc)
            self.n_labels = len(unique_tags)
            print(f"Unique Tags: {self.n_labels}\n", unique_tags)
            self.tag2id = {tag: i for i, tag in enumerate(unique_tags)}
            self.id2tag = {i: tag for tag, i in self.tag2id.items()}

        return self._to_dataset(seqs, labels, disorder)

    def _to_dataset(self, seqs, labels, disorder) -> 'SSPDataset':
        seqs_encodings = self.tokenizer(seqs, is_split_into_words=True,
                                        return_offsets_mapping=True,
                                        max_length=self.max_length,
                                        pad_to_max_length=True,
                                        truncation=True, padding=True)

        labels_encodings = self.encode_tags(labels, seqs_encodings)
        mask_disorder(labels_encodings, disorder)
        _ = seqs_encodings.pop("offset_mapping")

        return SSPDataset(seqs_encodings, labels_encodings)

    def model(self):
        if self._model is None:
            try:
                self._model = AutoModelForTokenClassification.from_pretrained(self.results_dir,
                                                                              num_labels=self.n_labels,
                                                                              id2label=self.id2tag,
                                                                              label2id=self.tag2id,
                                                                              gradient_checkpointing=False)

            except Exception as e:
                print(f"Failure loading model from {self.results_dir}. It probably doesn't exist yet.")
                print(e, f"Loading model from {self.model_name}...")
                if self.eval_mode:
                    raise e
                self._model = AutoModelForTokenClassification.from_pretrained(self.model_name,
                                                                              num_labels=self.n_labels,
                                                                              id2label=self.id2tag,
                                                                              label2id=self.tag2id,
                                                                              gradient_checkpointing=False)
        return self._model

    def do_training(self, train_data, val_data, model=None):

        if model is None:
            model = self.model
        trainer = self.get_trainer(model, train_dataset=train_data,
                                   val_dataset=val_data)
        trainer.train()
        if trainer.tokenizer is None:
            trainer.tokenizer = self.tokenizer
        trainer.save_model(self.results_dir)
        return trainer

    def align_predictions(self, predictions: np.ndarray, label_ids: np.ndarray):
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(self.id2tag[label_ids[i][j]])
                    preds_list[i].append(self.id2tag[preds[i][j]])

        return preds_list, out_label_list

    def compute_metrics(self, p: EvalPrediction):
        preds_list, out_label_list = self.align_predictions(p.predictions, p.label_ids)
        return {
            "accuracy": accuracy_score(out_label_list, preds_list),
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }
