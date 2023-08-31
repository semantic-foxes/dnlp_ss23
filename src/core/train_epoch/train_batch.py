from __future__ import annotations
import torch
from torch import nn


def _batch_forward_sentiment(batch, model, device, criterion):
    ids, attention_masks = batch['token_ids'], batch['attention_masks']

    ids = ids.to(device)
    attention_masks = attention_masks.to(device)

    predictions = model('sentiment', ids, attention_masks)

    loss = criterion(predictions, batch['targets']).sum()
    loss.backward()

    return predictions


def _batch_forward_similarity_standard(batch, model, task, device, criterion):
    ids_1, attention_masks_1, ids_2, attention_masks_2 = \
        (batch['token_ids_1'], batch['attention_masks_1'],
         batch['token_ids_2'], batch['attention_masks_2'])

    ids_1 = ids_1.to(device)
    ids_2 = ids_2.to(device)
    attention_masks_1 = attention_masks_1.to(device)
    attention_masks_2 = attention_masks_2.to(device)

    predictions = model(task, ids_1, attention_masks_1, ids_2, attention_masks_2)

    loss = criterion(predictions, batch['targets']).sum()
    loss.backward()

    return predictions


def _batch_forward_similarity_contrastive(batch_list, model, task, device, criterion):
    for batch in batch_list:
        ids_1, attention_masks_1, ids_2, attention_masks_2 = \
            (batch['token_ids_1'], batch['attention_masks_1'],
             batch['token_ids_2'], batch['attention_masks_2'])

        ids_1 = ids_1.to(device)
        ids_2 = ids_2.to(device)
        attention_masks_1 = attention_masks_1.to(device)
        attention_masks_2 = attention_masks_2.to(device)

        predictions = model(task, ids_1, attention_masks_1, ids_2, attention_masks_2)

        loss = criterion(predictions, batch['targets']).sum()
        loss.backward()
    # TODO: this is train only!
    #return predictions


def _batch_forward_similarity_triplet(batch, model, task, device):
    # Should this accept task or not?
    # Is this pretrain only?
    (token_ids_anchor,
     attention_masks_anchor,
     token_ids_positive,
     attention_masks_positive,
     token_ids_negative,
     attention_masks_negative) = (
        batch['token_ids_anchor'],
        batch['attention_masks_anchor'],
        batch['token_ids_positive'],
        batch['attention_masks_positive'],
        batch['token_ids_negative'],
        batch['attention_masks_negative'])

    (token_ids_anchor,
     attention_masks_anchor,
     token_ids_positive,
     attention_masks_positive,
     token_ids_negative,
     attention_masks_negative) = (
        token_ids_anchor.to(device),
        attention_masks_anchor.to(device),
        token_ids_positive.to(device),
        attention_masks_positive.to(device),
        token_ids_negative.to(device),
        attention_masks_negative.to(device))

    # TODO: Get BERT embeddings
    #embedding_anchor = model.embed(token_ids_anchor, attention_masks_anchor)
    #embedding_positive = model.embed(token_ids_positive, attention_masks_positive)
    #embedding_negative = model.embed(token_ids_negative, attention_masks_negative)
    # TODO: cosine distance
    #criterion = torch.nn.TripletMarginLoss()
    loss = criterion(embedding_anchor, embedding_positive, embedding_negative).sum()
    loss.backward()
    # TODO: this is train only!
    #return predictions

# TODO: remove
def _batch_forward(
        batch: torch.tensor,
        model: nn.Module,
        task: str,
        device: torch.device,
        criterion: torch.nn.Module,
        train_mode: str
):
    if task == 'sentiment':
        return _batch_forward_sentiment(batch, model, device, criterion)
    elif task == 'paraphrase_classifier' or task == 'paraphrase_regressor':
        if train_mode == 'contrastive':
            return _batch_forward_similarity_contrastive(batch, model, task, device, criterion)
        elif train_mode == 'triplet':
            return _batch_forward_similarity_triplet(batch, model, task, device)
        else:
            return _batch_forward_similarity_standard(batch, model, task, device, criterion)
    else:
        raise NotImplementedError

def train_one_batch_multitask(
        model: nn.Module,
        batch: torch.tensor,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        task: str,
        train_mode: str
):
    optimizer.zero_grad()
    # TODO: pass correct option instead of 'criterion'
    predictions = _batch_forward(batch, model, task, device, criterion, train_mode)
    optimizer.step()
