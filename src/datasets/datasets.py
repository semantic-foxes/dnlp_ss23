import pandas as pd
import torch
from torch.utils.data import Dataset

from tokenizer import BertTokenizer


class SSTDataset(Dataset):
    def __init__(
            self,
            dataset: pd.DataFrame,
            return_labels: bool = True,
            local_only: bool = False
    ):
        self.dataset = dataset['sentence']
        self.ids = list(dataset.index)

        self.return_labels = return_labels
        if return_labels:
            self.labels = dataset['label']

        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path='bert-base-uncased',
            local_files_only=local_only
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.return_labels:
            return self.dataset[index], self.labels[index]
        else:
            return self.dataset[index]

    def pad_data(self, data):
        if self.return_labels:
            sentences = [x[0] for x in data]
            labels = [x[1] for x in data]
        else:
            sentences = data

        encoding = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        if self.return_labels:
            labels = torch.LongTensor(labels)
            return token_ids, attention_mask, labels
        else:
            return token_ids, attention_mask

    def collate_fn(self, batch_data):
        if self.return_labels:
            token_ids, attention_mask, labels = self.pad_data(batch_data)
            batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

        else:
            token_ids, attention_mask = self.pad_data(batch_data)
            batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
            }

        return batched_data
