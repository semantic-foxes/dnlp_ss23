#!/usr/bin/env python3

"""
This module contains our Dataset classes and functions to load the 3 datasets we're using.

You should only need to call load_multitask_data to get the training and dev examples
to train your model.
"""


import csv

import torch
from torch.utils.data import Dataset
from tokenizer import BertTokenizer


def preprocess_string(s):
    return " ".join(
        s.lower()
        .replace(".", " .")
        .replace("?", " ?")
        .replace(",", " ,")
        .replace("'", " '")
        .split()
    )


class SentencePairDataset(Dataset):
    def __init__(self, dataset, args, isRegression=False):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(
            sent1, return_tensors="pt", padding=True, truncation=True
        )
        encoding2 = self.tokenizer(
            sent2, return_tensors="pt", padding=True, truncation=True
        )

        token_ids = torch.LongTensor(encoding1["input_ids"])
        attention_mask = torch.LongTensor(encoding1["attention_mask"])
        token_type_ids = torch.LongTensor(encoding1["token_type_ids"])

        token_ids2 = torch.LongTensor(encoding2["input_ids"])
        attention_mask2 = torch.LongTensor(encoding2["attention_mask"])
        token_type_ids2 = torch.LongTensor(encoding2["token_type_ids"])
        if self.isRegression:
            labels = torch.DoubleTensor(labels)
        else:
            labels = torch.LongTensor(labels)

        return (
            token_ids,
            token_type_ids,
            attention_mask,
            token_ids2,
            token_type_ids2,
            attention_mask2,
            labels,
            sent_ids,
        )

    def collate_fn(self, all_data):
        (
            token_ids,
            token_type_ids,
            attention_mask,
            token_ids2,
            token_type_ids2,
            attention_mask2,
            labels,
            sent_ids,
        ) = self.pad_data(all_data)

        batched_data = {
            "token_ids_1": token_ids,
            "token_type_ids_1": token_type_ids,
            "attention_mask_1": attention_mask,
            "token_ids_2": token_ids2,
            "token_type_ids_2": token_type_ids2,
            "attention_mask_2": attention_mask2,
            "labels": labels,
            "sent_ids": sent_ids,
        }

        return batched_data


class SentencePairTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding1 = self.tokenizer(
            sent1, return_tensors="pt", padding=True, truncation=True
        )
        encoding2 = self.tokenizer(
            sent2, return_tensors="pt", padding=True, truncation=True
        )

        token_ids = torch.LongTensor(encoding1["input_ids"])
        attention_mask = torch.LongTensor(encoding1["attention_mask"])
        token_type_ids = torch.LongTensor(encoding1["token_type_ids"])

        token_ids2 = torch.LongTensor(encoding2["input_ids"])
        attention_mask2 = torch.LongTensor(encoding2["attention_mask"])
        token_type_ids2 = torch.LongTensor(encoding2["token_type_ids"])

        return (
            token_ids,
            token_type_ids,
            attention_mask,
            token_ids2,
            token_type_ids2,
            attention_mask2,
            sent_ids,
        )

    def collate_fn(self, all_data):
        (
            token_ids,
            token_type_ids,
            attention_mask,
            token_ids2,
            token_type_ids2,
            attention_mask2,
            sent_ids,
        ) = self.pad_data(all_data)

        batched_data = {
            "token_ids_1": token_ids,
            "token_type_ids_1": token_type_ids,
            "attention_mask_1": attention_mask,
            "token_ids_2": token_ids2,
            "token_type_ids_2": token_type_ids2,
            "attention_mask_2": attention_mask2,
            "sent_ids": sent_ids,
        }

        return batched_data


def load_multitask_test_data():
    paraphrase_data = []
    with open(paraphrase_filename, "r", encoding="utf-8") as fp:
        for record in csv.DictReader(fp, delimiter="\t"):
            # if record['split'] != split:
            #    continue
            paraphrase_data.append(
                (
                    preprocess_string(record["sentence1"]),
                    preprocess_string(record["sentence2"]),
                )
            )

    print(f"Loaded {len(paraphrase_data)} test examples from {paraphrase_filename}")

    similarity_data = []
    with open(similarity_filename, "r", encoding="utf-8") as fp:
        for record in csv.DictReader(fp, delimiter="\t"):
            similarity_data.append(
                (
                    preprocess_string(record["sentence1"]),
                    preprocess_string(record["sentence2"]),
                )
            )

    print(f"Loaded {len(similarity_data)} test examples from {similarity_filename}")

    return sentiment_data, paraphrase_data, similarity_data


def load_multitask_data(
    sentiment_filename, paraphrase_filename, similarity_filename, split="train"
):
    paraphrase_data = []
    if split == "test":
        with open(paraphrase_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent_id = record["id"].lower().strip()
                paraphrase_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        sent_id,
                    )
                )

    else:
        with open(paraphrase_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                try:
                    sent_id = record["id"].lower().strip()
                    paraphrase_data.append(
                        (
                            preprocess_string(record["sentence1"]),
                            preprocess_string(record["sentence2"]),
                            int(float(record["is_duplicate"])),
                            sent_id,
                        )
                    )
                except:
                    pass

    print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")

    similarity_data = []
    if split == "test":
        with open(similarity_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent_id = record["id"].lower().strip()
                similarity_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        sent_id,
                    )
                )
    else:
        with open(similarity_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent_id = record["id"].lower().strip()
                similarity_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        float(record["similarity"]),
                        sent_id,
                    )
                )

    print(f"Loaded {len(similarity_data)} {split} examples from {similarity_filename}")

    return sentiment_data, num_labels, paraphrase_data, similarity_data
