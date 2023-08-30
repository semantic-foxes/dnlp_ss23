from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from tokenizer import BertTokenizer


tqdm.pandas()


class SSTDataset(Dataset):
    def __init__(
            self,
            dataset_path: str,
            return_targets: bool = True,
            local_only: bool = False,
            nrows: int = None
    ):
        dataset = pd.read_csv(dataset_path, index_col=0, delimiter='\t', nrows=nrows)

        # Data handling
        dataset.index = dataset['id']
        dataset.drop('id', axis=1, inplace=True)
        dataset['sentence'] = dataset['sentence'].str.lower().str.strip()

        self.dataset = dataset['sentence']
        self.ids = list(dataset.index)

        self.return_targets = return_targets
        if return_targets:
            self.targets = dataset['sentiment']

        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path='bert-base-uncased',
            local_files_only=local_only
        )

        self.task = 'sentiment'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.return_targets:
            return self.dataset.iloc[index], self.targets.iloc[index]
        else:
            return self.dataset.iloc[index]

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
            dataset_path: str,
            binary_task: bool = True,
            return_targets: bool = True,
            local_only: bool = False,
            index_col: int = 0,
            nrows: int = None,
            inflation_params: tuple[int, int] = None,
    ):
        dataset = pd.read_csv(dataset_path, index_col=index_col, delimiter='\t', nrows=nrows)
        # Data handling
        dataset.index = dataset['id']
        dataset.drop('id', axis=1, inplace=True)
        dataset.dropna(inplace=True)
        self.dataset = dataset.loc[:, ['sentence1', 'sentence2']]
        self.dataset['sentence1'] = self.dataset['sentence1'] \
            .progress_apply(self.preprocess_string)
        self.dataset['sentence2'] = self.dataset['sentence2'] \
            .progress_apply(self.preprocess_string)
        self.ids = list(dataset.index)

        self.binary_task = binary_task
        if binary_task:
            self.task = 'paraphrase_classifier'
        else:
            self.task = 'paraphrase_regressor'

        self.return_targets = return_targets
        if return_targets:
            if self.binary_task:
                self.targets = dataset['is_duplicate'].astype(float).astype(int)
            else:
                self.targets = dataset['similarity'].astype(float)

        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path='bert-base-uncased',
            local_files_only=local_only
        )
        if inflation_params is not None:
            self.inflation = True
            self.batch_size = inflation_params[0]
            self.exp_factor = inflation_params[1]
            self.inflation_idx = self._compute_inflation_idx(self.batch_size, self.exp_factor)
        else:
            self.inflation = False

    @staticmethod
    def preprocess_string(s):
        return ' '.join(s.lower()
                        .replace('.', ' .')
                        .replace('?', ' ?')
                        .replace(',', ' ,')
                        .replace('\'', ' \'')
                        .split())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.return_targets:
            result = (self.dataset.iloc[index]['sentence1'],
                    self.dataset.iloc[index]['sentence2'],
                    self.targets.iloc[index])
            return result
        else:
            return (self.dataset.iloc[index]['sentence1'],
                    self.dataset.iloc[index]['sentence2'])

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

        result = {
            'token_ids_1': token_ids_1,
            'token_type_ids_1': token_type_ids_1,
            'attention_masks_1': attention_masks_1,
            'token_ids_2': token_ids_2,
            'token_type_ids_2': token_type_ids_2,
            'attention_masks_2': attention_masks_2,
        }

        if self.return_targets:
            if self.binary_task:
                result['targets'] = torch.LongTensor(targets)
            else:
                result['targets'] = torch.FloatTensor(targets)
        if self.inflation:
            result = self._inflate_batch(result, self.inflation_idx, self.return_targets)

        return result

    @staticmethod
    def _compute_inflation_idx(batch_size: int, exp_factor: int) -> np.ndarray:
        inflation_idx = (np.repeat(batch_size * np.arange(batch_size), exp_factor)
                          + (np.add.outer(np.arange(exp_factor), np.arange(batch_size))
                             % np.repeat(batch_size, (batch_size))
                            ).T.flatten())
        return inflation_idx

    @staticmethod
    def _inflate_batch(batch: dict, inflation_idx: np.ndarray = None, return_targets: bool = True) -> dict:
        result = {}
        batch_size = batch['token_ids_1'].shape[0]
        if inflation_idx is not None:
            def apply_filter(x: torch.tensor) -> torch.tensor:
                return x[inflation_idx]
        else:
            def apply_filter(x):
                return x
        for k in ('token_ids_1', 'token_type_ids_1', 'attention_masks_1'):
            result[k] = apply_filter(batch[k].repeat(batch_size, 1))
        for k in ('token_ids_2', 'token_type_ids_2', 'attention_masks_2'):
            result[k] = apply_filter(batch[k].repeat_interleave(batch_size, dim=0))
        if return_targets:
            result['targets'] = apply_filter(torch.diag(batch['targets']).flatten())
        return result

    @classmethod
    def inflate_batch(cls, batch: dict, exp_factor: int = None, return_targets: bool = True) -> dict:
        """ Transform a batch of pairs (a_i, b_i) by adding to it some of the pairs (a_i, b_j) for j != i.
        The resulting batch will be exp_factor times bigger than the original one. The default value of exp_factor
        corresponds to adding all pairs (a_i, b_j), and is equivalent to exp_factor=batch_size.
        Should be applied after collate_fn.
        """
        batch_size = batch['token_ids_1'].shape[0]
        inflation_idx = cls._compute_inflation_idx(batch_size, exp_factor)
        result = cls._inflate_batch(batch, inflation_idx, return_targets)
