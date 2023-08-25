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
            local_only: bool = False
    ):
        dataset = pd.read_csv(dataset_path, index_col=0, delimiter='\t')

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
    ):
        dataset = pd.read_csv(dataset_path, index_col=0, delimiter='\t')

        # Data handling
        dataset.index = dataset['id']
        dataset.drop('id', axis=1, inplace=True)
        dataset.dropna(inplace=True) # TODO: Modify for a correct test behaviour
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
