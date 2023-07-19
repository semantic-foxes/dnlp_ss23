import os
import random
from types import SimpleNamespace
import argparse
import csv
import yaml

import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from tokenizer import BertTokenizer
from optimizer import AdamW
from src.utils import seed_everything, get_device, logger, generate_device
from src.models import BertSentimentClassifier

TQDM_DISABLE = False


class SentimentDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=args.local_files_only)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sents': sents,
            'sent_ids': sent_ids
        }

        return batched_data


class SentimentTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=args.local_files_only)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids = self.pad_data(all_data)

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'sents': sents,
            'sent_ids': sent_ids
        }

        return batched_data


# Load the data: a list of (sentence, label)
def load_data(filename, flag='train'):
    num_labels = {}
    data = []
    if flag == 'test':
        with open(filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                data.append((sent, sent_id))

    else:
        with open(filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                data.append((sent, label, sent_id))
        print(f"load {len(data)} data from {filename}")

    if flag == 'train':
        return data, len(num_labels)
    else:
        return data


# Evaluate the model for accuracy.
@torch.no_grad()
def model_eval(model, eval_dataloader, device):

    model.eval()
    y_true = []
    y_pred = []
    sents = []
    sent_ids = []
    for batch in tqdm(eval_dataloader, desc=f'eval', leave=False, disable=TQDM_DISABLE):
        b_ids, b_mask, b_labels, b_sents, b_sent_ids = \
            batch['token_ids'], batch['attention_mask'], batch['labels'], \
            batch['sents'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask).detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sents, sent_ids


@torch.no_grad()
def generate_predictions(
        model,
        dataloader,
        device,
        verbose=False
):
    model.eval()
    y_pred = []
    sents = []
    sent_ids = []
    progress_bar = tqdm(dataloader, desc='Test predictions', disable=verbose)

    for batch in progress_bar:
        ids, attention_masks, sents, sent_ids = \
            batch['token_ids'], batch['attention_mask'],\
            batch['sents'], batch['sent_ids']

        ids = ids.to(device)
        attention_masks = attention_masks.to(device)

        predicted_logits = model(ids, attention_masks)
        predicted_logits = predicted_logits.cpu().numpy()
        predictions = np.argmax(predicted_logits, axis=1).flatten()

        y_pred.extend(predictions)
        # TODO: Do we even need this?
        sents.extend(sents)
        sent_ids.extend(sent_ids)

    return y_pred, sents, sent_ids


def save_model(model, optimizer, config, filepath):

    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)

    logger.info(f'Saving the model to {filepath}.')

    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    logger.info(f'Successfully saved the model to {filepath}.')


# TODO: custom criterion not supported yet.
def train_one_epoch(
        model,
        train_dataloader,
        optimizer,
        criterion,
        device,
        verbose: bool = True,
        current_epoch: int = None,
) -> float:
    model.train()

    if verbose:
        if current_epoch is not None:
            pbar = tqdm(train_dataloader, leave=False,
                        desc=f'Training epoch {current_epoch}')
        else:
            pbar = tqdm(train_dataloader, leave=False,
                        desc=f'Training model')
    else:
        pbar = train_dataloader

    train_loss = 0

    for batch in pbar:
        ids, attention_masks, labels = \
            batch['token_ids'], batch['attention_mask'], batch['labels']

        ids = ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(ids, attention_masks)

        # TODO: This is an incorrect way to calculate this.
        loss = F.cross_entropy(logits, labels.view(-1), reduction='sum') / len(batch)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss / len(train_dataloader)
    return train_loss


def train_and_validate(
        model,
        train_dataloader,
        val_dataloader,
        n_epochs,
        optimizer,
        criterion,
        device,
        save_best_path: str = None
):
    model = model.to(device)
    best_val_acc = 0

    # Run for the specified number of epochs
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            verbose=True,
            current_epoch=epoch,
        )

        train_acc, train_f1, *_ = model_eval(model, train_dataloader, device)
        val_acc, val_f1, *_ = model_eval(model, val_dataloader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, optimizer, CONFIG, save_best_path)

        message = (f'Finished training epoch {epoch}. Train loss: {train_loss:.3f}, '
                   f'train accuracy: {train_acc:.3f}, val accuracy: {val_acc:.3f}.')
        logger.info(message)

    message = f'Finished training and validation the model.'
    logger.info(message)


def evaluate(
        model,
        val_dataloader,
        test_dataloader,
        device,
):

    config = saved['model_config']
    model = model.to(device)

    dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(dev_dataloader, model, device)
    print('DONE DEV')
    test_pred, test_sents, test_sent_ids = generate_predictions(test_dataloader, model, device)
    print('DONE Test')

    # Creating submissions
    with open(args.dev_out, "w+") as f:
        print(f"dev acc :: {dev_acc :.3f}")
        f.write(f"id \t Predicted_Sentiment \n")
        for p, s in zip(dev_sent_ids, dev_pred):
            f.write(f"{p} , {s} \n")

    with open(args.test_out, "w+") as f:
        f.write(f"id \t Predicted_Sentiment \n")
        for p, s in zip(test_sent_ids, test_pred):
            f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="finetune")
    parser.add_argument("--use_gpu", action='store_true', default='True')
    parser.add_argument("--dev_out", type=str, default="sst-dev-out.csv")
    parser.add_argument("--test_out", type=str, default="sst-test-out.csv")

    parser.add_argument("--batch_size", help='sst: 64 can fit a 12GB GPU', type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    parser.add_argument("--local_files_only", action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)

    with open('config.yaml', 'r') as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)

    config_data = CONFIG['data']
    config_bert = CONFIG['bert_model']
    config_train = CONFIG['train']

    device = generate_device(CONFIG['use_cuda'])

    # Create datasets
    train_data, num_labels = load_data(config_data['dataset']['train_path'])
    val_data = load_data(config_data['dataset']['val_path'], flag='val')

    train_dataset = SentimentDataset(train_data)
    val_dataset = SentimentDataset(val_data)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        batch_size=config_data['dataloader']['batch_size'],
        num_workers=config_data['dataloader']['num_workers'],
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        batch_size=config_data['dataloader']['batch_size'],
        num_workers=config_data['dataloader']['num_workers'],
    )

    # Create a model
    model_config = SimpleNamespace(**{
        'num_labels': num_labels,
        'option': config_bert['mode'],
        'hidden_size': config_bert['hidden_size'],
        'hidden_dropout_prob': config_bert['dropout_prob'],
        'data_dir': '.',
        'local_files_only': config_bert['local_files_only']
    })
    model = BertSentimentClassifier(model_config)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config_train['lr'])

    logger.info(f'Starting training the {config_bert["mode"]} BERT model.')

    train_and_validate(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        n_epochs=config_train['n_epochs'],
        optimizer=optimizer,
        criterion=None,
        device=device,
        save_best_path=config_train['checkpoint_path']
    )
