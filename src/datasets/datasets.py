import pandas as pd
import torch
from torch.utils.data import Dataset

from tokenizer import BertTokenizer


class SSTDataset(Dataset):
    def __init__(
            self,
            dataset: pd.DataFrame,
            return_targets: bool = True,
            local_only: bool = False
    ):
        self.dataset = dataset['sentence']
        self.ids = list(dataset.index)

        self.return_targets = return_targets
        if return_targets:
            self.targets = dataset['sentiment']

        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path='bert-base-uncased',
            local_files_only=local_only
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.return_targets:
            return self.dataset[index], self.targets[index]
        else:
            return self.dataset[index]

    def collate_fn(self, batch_data):
        if self.return_targets:
            sentences = [x[0] for x in batch_data]
            targets = [x[1] for x in batch_data]
        else:
            sentences = batch_data

        encodings = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encodings['input_ids'])
        attention_masks = torch.LongTensor(encodings['attention_mask'])

        result = {
            'token_ids': token_ids,
            'attention_masks': attention_masks,
        }

        if self.return_targets:
            result['targets'] = torch.LongTensor(targets)

        return result


class SentenceSimilarityDataset(Dataset):
    def __init__(
            self,
            dataset: pd.DataFrame,
            binary_task: bool = True,
            return_targets: bool = True,
            local_only: bool = False,
    ):
        self.dataset = dataset
        self.dataset = dataset[['sentence1', 'sentence2']]
        self.ids = list(dataset.index)

        self.binary_task = binary_task
        self.return_targets = return_targets

        if return_targets:
            if self.binary_task:
                self.targets = dataset['is_duplicate']
            else:
                self.targets = dataset['similarity']

        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path='bert-base-uncased',
            local_files_only=local_only
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.return_targets:
            return self.dataset[index], self.targets[index]
        else:
            return self.dataset[index]

    def collate_fn(self, batch_data):
        sentences_1 = [x[0] for x in batch_data]
        sentences_2 = [x[1] for x in batch_data]

        if self.return_targets:
            targets = [x[2] for x in batch_data]

        encodings_1 = self.tokenizer(
            sentences_1,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        token_ids_1 = torch.LongTensor(encodings_1['input_ids'])
        attention_masks_1 = torch.LongTensor(encodings_1['attention_mask'])
        token_type_ids_1 = torch.LongTensor(encodings_1['token_type_ids'])

        encodings_2 = self.tokenizer(
            sentences_2,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        token_ids_2 = torch.LongTensor(encodings_2['input_ids'])
        attention_masks_2 = torch.LongTensor(encodings_2['attention_mask'])
        token_type_ids_2 = torch.LongTensor(encodings_2['token_type_ids'])

        # TODO: might need to cast for the binary task to DoubleTensor

        # if self.binary_task:
        #     targets = torch.DoubleTensor(targets)
        # else:
        #     targets = torch.LongTensor(targets)

        result = {
            'token_ids_1': token_ids_1,
            'token_type_ids_1': token_type_ids_1,
            'attention_masks_1': attention_masks_1,
            'token_ids_2': token_ids_2,
            'token_type_ids_2': token_type_ids_2,
            'attention_masks_2': attention_masks_2,
        }

        if self.return_targets:
            result['targets'] = torch.LongTensor(targets)

        return result