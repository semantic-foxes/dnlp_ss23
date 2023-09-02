from __future__ import annotations
from functools import partial
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
    ):
        dataset = pd.read_csv(dataset_path, index_col=index_col, delimiter='\t', nrows=nrows)
        # Data handling
        dataset.index = dataset['id']
        dataset.drop('id', axis=1, inplace=True)
        dataset.dropna(inplace=True)
        self.dataset = dataset[['sentence1', 'sentence2']]
        self.dataset.loc[:, 'sentence1'] = self.dataset['sentence1'].apply(self.preprocess_string)
        self.dataset.loc[:, 'sentence2'] = self.dataset['sentence2'].apply(self.preprocess_string)
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

    def _encode(self, sentences: list[str], padding = True, max_length=None
               ) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        encodings = self.tokenizer(
            sentences,
            return_tensors='pt',
            padding=padding,
            max_length=max_length,
            truncation=True
        )
        token_ids = torch.LongTensor(encodings['input_ids'])
        attention_masks = torch.LongTensor(encodings['attention_mask'])
        token_type_ids = torch.LongTensor(encodings['token_type_ids'])
        return token_ids, attention_masks, token_type_ids

    @staticmethod
    def _mask(dropout_rate, token_ids, attention_masks, token_type_ids):
            # TODO: special_tokens_mask (IMPORTANT)
            # (add &~special_tokens_mask; see DataCollatorForLanguageModeling)
            dropout_mask = torch.empty(
                token_ids.shape, device=token_ids.device, dtype=int
            ).bernoulli_(1 - dropout_rate)

            # (c) Lingyu Zhang
            def mask_shift(tensor, mask):
                masked = tensor * mask
                shifted = torch.gather(
                    masked, 1,
                    masked.ne(0).argsort(dim=1, descending=True, stable=True)
                )
                return shifted
            token_ids = mask_shift(token_ids, dropout_mask)
            attention_masks = mask_shift(attention_masks, dropout_mask)
            token_type_ids = mask_shift(token_type_ids, dropout_mask)
            return token_ids, attention_masks, token_type_ids

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

        return result

    def _collate_fn_contrastive(self, exp_factor, batch_data):
        batch = self.collate_fn(batch_data)
        result = [batch]

        for i in range(1, exp_factor):
            batch_transformed = {}
            for k in ('token_ids_1', 'token_type_ids_1', 'attention_masks_1'):
                batch_transformed[k] = batch[k]
            for k in ('token_ids_2', 'token_type_ids_2', 'attention_masks_2'):
                batch_transformed[k] = batch[k].roll(-i)
            if self.return_targets:
                batch_transformed['targets'] = torch.zeros_like(batch['targets'])
            result.append(batch_transformed)

        return result

    def collate_fn_contrastive(self, exp_factor):
        """Create a collate_fn function for contrastive learning.
        The function returns a list of exp_factor batches. The first
        element is the original batch, the other elements are
        additional batches of negative pairs, with targets set to 0 if
        return_targets is set. For batch structure, see collate_fn().
        """
        return partial(self._collate_fn_contrastive, exp_factor)

    def _collate_fn_triplet_unsupervised(self, dropout_rate, batch_data):
        sentences_1 = np.array([x[0] for x in batch_data], dtype=object)
        
        def mask_encode(sentences):
            return self._mask(dropout_rate, *self._encode(sentences.tolist()))
        
        # TODO: rewrite to return tuple, remove code triplication
        (token_ids_anchor,
         attention_masks_anchor,
         token_type_ids_anchor) = mask_encode(sentences_1)

        (token_ids_positive,
         attention_masks_positive,
         token_type_ids_positive) = mask_encode(sentences_1)

        # TODO: May be slower than rolling after mask_encode
        (token_ids_negative,
         attention_masks_negative,
         token_type_ids_negative) = mask_encode(np.roll(sentences_1, -1))

        result = {
            'token_ids_anchor': token_ids_anchor,
            'attention_masks_anchor': attention_masks_anchor,
            'token_type_ids_anchor': token_type_ids_anchor,
            'token_ids_positive': token_ids_positive,
            'attention_masks_positive': attention_masks_positive,
            'token_type_ids_positive': token_type_ids_positive,
            'token_ids_negative': token_ids_negative,
            'attention_masks_negative': attention_masks_negative,
            'token_type_ids_negative': token_type_ids_negative,
        }

        return result

    def collate_fn_triplet_unsupervised(self, dropout_rate):
        """Returns triplets of the form:
        anchor = dropout(sentences_1),
        positive = dropout(sentences_1),
        negative = roll(dropout(sentences_1)).
        For batch structure, see collate_fn().
        """
        return partial(self._collate_fn_triplet_unsupervised, dropout_rate)

    def _collate_fn_triplet_supervised_v1(self, dropout_rate, batch_data):
        # TODO:
        # for positive pairs pos = sentences_2, neg = roll.
        # for negative pairs pos = dropout(sentences_1), neg = roll.
        pass

    def _collate_fn_triplet(self, dropout_rate, batch_data):
        sentences_1 = np.array([x[0] for x in batch_data], dtype=object)
        sentences_2 = np.array([x[1] for x in batch_data], dtype=object)

        if self.return_targets:
            targets = [x[2] for x in batch_data]
            if self.binary_task:
                targets = torch.LongTensor(targets)
            else:
                targets = torch.FloatTensor(targets)
        else:
            raise NotImplementedError()
        
        def mask_encode(sentences):
            return self._mask(dropout_rate, *self._encode(sentences.tolist(),
                                                          padding='max_length', max_length=40))

        # TODO: slow
        (token_ids_anchor,
         attention_masks_anchor,
         token_type_ids_anchor) = self._encode(sentences_1.tolist(),
                                               padding='max_length', max_length=40)

        sentences_1_masked_encoded = mask_encode(sentences_1)
        sentences_2_encoded = self._encode(sentences_2.tolist(),
                                           padding='max_length', max_length=40)
        # TODO: slow
        sentences_2_roll = mask_encode(np.roll(sentences_1, -1))
        
        target_mask = (targets != 0).unsqueeze(1)
        
        result_positive = [target_mask * sentences_2_encoded[i] 
                           + ~target_mask * sentences_1_masked_encoded[i]
                           for i in range(3)]
        result_negative = [target_mask * sentences_2_roll[i]
                           + ~target_mask * sentences_2_encoded[i] 
                           for i in range(3)]
        
        result = {
            'token_ids_anchor': token_ids_anchor,
            'attention_masks_anchor': attention_masks_anchor,
            'token_type_ids_anchor': token_type_ids_anchor,
            'token_ids_positive': result_positive[0],
            'attention_masks_positive': result_positive[1],
            'token_type_ids_positive': result_positive[2],
            'token_ids_negative': result_negative[0],
            'attention_masks_negative': result_negative[1],
            'token_type_ids_negative': result_negative[2]
        }
        
        return result

    def collate_fn_triplet(self, dropout_rate):
        """ Returns triplets of the following form:
        anchor = sentences_1
        positive = sentences_2 for positive pairs, 
                   dropout(sentences_1) for negative pairs
        negative = roll(sentences_2) for positive pairs
                   sentences_2 for negative pairs
        """
        return partial(self._collate_fn_triplet, dropout_rate)