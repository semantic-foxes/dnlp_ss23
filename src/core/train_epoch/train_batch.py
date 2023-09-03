from __future__ import annotations
from typing import Union
import torch
from torch import nn


def _batch_forward_sentiment(
        batch: torch.tensor,
        device: torch.device,
        model: nn.Module,
        task: str = 'sentiment',
) -> torch.tensor:
    ids, attention_masks = batch['token_ids'], batch['attention_masks']

    ids = ids.to(device)
    attention_masks = attention_masks.to(device)

    predictions = model(task, ids, attention_masks)
    return predictions

def _batch_forward_similarity(
        batch: torch.tensor,
        device: torch.device,
        model: nn.Module,
        task: str,
) -> torch.tensor:
    ids_1, attention_masks_1, ids_2, attention_masks_2 = \
        (batch['token_ids_1'], batch['attention_masks_1'],
         batch['token_ids_2'], batch['attention_masks_2'])

    ids_1 = ids_1.to(device)
    ids_2 = ids_2.to(device)
    attention_masks_1 = attention_masks_1.to(device)
    attention_masks_2 = attention_masks_2.to(device)

    if task == 'paraphrase_classifier':
        # predictions with embeddings
        return model(task, ids_1, attention_masks_1, ids_2, attention_masks_2, True)

    predictions = model(task, ids_1, attention_masks_1, ids_2, attention_masks_2)
    return (predictions, [])

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
        task: str,
        cosine_loss = None,
):
    if task == 'sentiment':
        predictions = _batch_forward_sentiment(batch, device, model)
    elif task == 'paraphrase_classifier' or task == 'paraphrase_regressor':
        predictions, embeddings = _batch_forward_similarity(batch, device, model, task)
    else:
        raise NotImplementedError

    targets = batch['targets'].to(device)

    loss = criterion(predictions, targets).sum()

    if cosine_loss and task == 'paraphrase_classifier':
        # targets should be -1,1
        loss = loss + cosine_loss(*embeddings, 2*targets-1) / len(targets) / 4 
    return loss

def train_one_batch_contrastive(
        batch_list: list[torch.tensor],
        criterion: torch.nn.Module,
        device: torch.device,
        model: nn.Module,
        task: str,
        weight: float = 0.5,
):
    weight_original = 1 - weight
    weight_extra = weight / (len(batch_list) - 1)

    original_batch = batch_list.pop(0)
    original_predictions, _ = _batch_forward_similarity(original_batch, device, model, task)
    targets = original_batch['targets'].to(device)
    loss = weight_original * criterion(original_predictions, targets).sum()

    for batch in batch_list:
        predictions, _ = _batch_forward_similarity(batch, device, model, task)
        targets = batch['targets'].to(device)
        loss += weight_extra * criterion(predictions, targets).sum()
    
    return loss

def train_one_batch_triplet(
        batch: torch.tensor,
        criterion: torch.nn.Module,
        device: torch.device,
        model: nn.Module,
        task: str,
        weight: float = 0.5
):
    label_map = {
        'token_ids_anchor': 'token_ids_1',
        'attention_masks_anchor': 'attention_masks_1',
        'token_type_ids_anchor': 'token_type_ids_1',
        'token_ids_positive': 'token_ids_2',
        'attention_masks_positive': 'attention_masks_2',
        'token_type_ids_positive': 'token_type_ids_2',
        'token_ids_negative': 'token_ids_2',
        'attention_masks_negative': 'attention_masks_2',
        'token_type_ids_negative': 'token_type_ids_2'
    }
    batch_positive = {label_map[k]: batch[k] for k in (
        'token_ids_anchor', 'attention_masks_anchor','token_type_ids_anchor',
        'token_ids_positive', 'attention_masks_positive', 'token_type_ids_positive')
                }
    batch_negative = {label_map[k]: batch[k] for k in (
        'token_ids_anchor', 'attention_masks_anchor','token_type_ids_anchor',
        'token_ids_negative', 'attention_masks_negative', 'token_type_ids_negative')
                }

    batch_size = len(batch['token_ids_anchor'])

    if task == 'paraphrase_classifier':
        dtype = torch.long
    else:
        dtype = torch.float

    predictions_positive, _ = _batch_forward_similarity(
        batch_positive, device, model, task)
    positive_loss = criterion(
        predictions_positive,
        torch.ones(batch_size, device=device, dtype=dtype)).sum()

    predictions_negative, _ = _batch_forward_similarity(
        batch_negative, device, model, task)
    negative_loss = criterion(
        predictions_negative,
        torch.zeros(batch_size, device=device, dtype=dtype)).sum()

    embeddings = _batch_forward_triplet(batch, device, model)
    triplet_criterion = torch.nn.TripletMarginLoss()
    triplet_loss = triplet_criterion(*embeddings).sum()

    loss = (weight * triplet_loss
            + 0.5 * (1 - weight) * positive_loss
            + 0.5 * (1 - weight) * negative_loss)
    return loss


def train_one_batch_multitask(
        model: nn.Module,
        batch: Union[torch.tensor, list[torch.tensor]],
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        task: str,
        train_mode: str,
        overall_config: dict = None,
        is_optimizer_step: bool = True,
        loss_divisor: int = 1,
        cosine_loss = None,
):
    if train_mode == 'standard' or task == 'sentiment' or task == 'paraphrase_regressor':
        loss = train_one_batch_standard(batch, criterion, device, model, task, cosine_loss=cosine_loss)
    elif train_mode == 'contrastive':
        weight = overall_config['train'].get('contrastive_weight', 0.5)
        loss = train_one_batch_contrastive(batch, criterion, device, model, task, weight=weight)
    elif train_mode == 'triplet':
        weight = overall_config['train'].get('triplet_weight', 0.5)
        loss = train_one_batch_triplet(batch, criterion, device, model, task, weight=weight)
    else:
        raise NotImplementedError(f"train_mode={train_mode} is not supported")

    loss /= loss_divisor

    loss.backward()
    if is_optimizer_step:
        optimizer.step()
        optimizer.zero_grad()