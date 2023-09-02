from __future__ import annotations
from typing import Union
import torch
from torch import nn


def _batch_forward_sentiment(
        batch: torch.tensor,
        device: torch.device,
        model: nn.Module,
) -> torch.tensor:
    ids, attention_masks = batch['token_ids'], batch['attention_masks']

    ids = ids.to(device)
    attention_masks = attention_masks.to(device)

    predictions = model('sentiment', ids, attention_masks)
    return predictions


def _batch_forward_similarity(
        batch: torch.tensor,
        device: torch.device,
        model: nn.Module,
        task: str
) -> torch.tensor:
    ids_1, attention_masks_1, ids_2, attention_masks_2 = \
        (batch['token_ids_1'], batch['attention_masks_1'],
         batch['token_ids_2'], batch['attention_masks_2'])

    ids_1 = ids_1.to(device)
    ids_2 = ids_2.to(device)
    attention_masks_1 = attention_masks_1.to(device)
    attention_masks_2 = attention_masks_2.to(device)

    predictions = model(task, ids_1, attention_masks_1, ids_2, attention_masks_2)
    return predictions

# TODO: remove
def _batch_forward(
        batch: torch.tensor,
        model: nn.Module,
        task: str,
        device: torch.device,
):
    if task == 'sentiment':
        predictions = _batch_forward_sentiment(batch, device, model)
    elif task == 'paraphrase_classifier' or task == 'paraphrase_regressor':
        predictions = _batch_forward_similarity(batch, device, model, task)
    else:
        raise NotImplementedError
    return predictions


def _batch_forward_triplet(
        batch: torch.tensor,
        device: torch.device,
        model: nn.Module,
) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
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

    embedding_anchor = model('embed', token_ids_anchor, attention_masks_anchor)
    embedding_positive = model('embed', token_ids_positive, attention_masks_positive)
    embedding_negative = model('embed', token_ids_negative, attention_masks_negative)

    return embedding_anchor, embedding_positive, embedding_negative

def train_one_batch_standard(
        batch: torch.tensor,
        criterion: torch.nn.Module,
        device: torch.device,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        task: str,
):
    optimizer.zero_grad()

    if task == 'sentiment':
        predictions = _batch_forward_sentiment(batch, device, model)
    elif task == 'paraphrase_classifier' or task == 'paraphrase_regressor':
        predictions = _batch_forward_similarity(batch, device, model, task)
    else:
        raise NotImplementedError

    targets = batch['targets'].to(device)

    loss = criterion(predictions, targets).sum()
    loss.backward()

    optimizer.step()

def train_one_batch_contrastive(
        batch_list: list[torch.tensor],
        criterion: torch.nn.Module,
        device: torch.device,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        task: str,
        return_predictions: bool = False,
        weight: float = 0.5
):
    weight_original = 1 - weight
    weight_extra = weight / (len(batch_list) - 1)
    
    optimizer.zero_grad()
    
    original_batch = batch_list.pop(0)
    original_predictions = _batch_forward_similarity(original_batch, device, model, task)
    targets = original_batch['targets'].to(device)
    loss = weight_original * criterion(original_predictions, targets).sum()
    loss.backward()

    for batch in batch_list:
        predictions = _batch_forward_similarity(batch, device, model, task)
        targets = batch['targets'].to(device)
        loss = weight_extra * criterion(predictions, targets).sum()
        loss.backward()

    optimizer.step()

    if return_predictions:
        return original_predictions


def train_one_batch_triplet(
        batch: torch.tensor,
        criterion: torch.nn.Module,
        device: torch.device,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        task: str
):
    optimizer.zero_grad()

    embeddings = _batch_forward_triplet(batch, device, model)

    loss = criterion(*embeddings).sum()
    loss.backward()

    optimizer.step()


def train_one_batch_multitask(
        model: nn.Module,
        batch: Union[torch.tensor, list[torch.tensor]],
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        task: str,
        train_mode: str,
        overall_config: dict = None
):
    if train_mode == 'standard' or task == 'sentiment':
        train_one_batch_standard(batch, criterion, device, model,
                                 optimizer, task)
    elif train_mode == 'contrastive':
        weight = overall_config['train'].get('contrastive_weight', 0.5)
        train_one_batch_contrastive(batch, criterion, device, model,
                                    optimizer, task, weight=weight)
    elif train_mode == 'triplet':
        forced_criterion = torch.nn.TripletMarginLoss()
        train_one_batch_triplet(batch, forced_criterion, device, model,
                                optimizer, task)
    else:
        raise NotImplementedError(f"train_mode={train_mode} is not supported")
