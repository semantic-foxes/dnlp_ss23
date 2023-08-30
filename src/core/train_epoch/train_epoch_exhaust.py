import random
from typing import List, Union
import numpy as np

from tqdm import tqdm

import torch
from torch import nn

from src.core.train_epoch.train_batch import _batch_forward, train_one_batch_multitask

from src.utils import logger

def sample_task_from_pool(
        dataloaders: List[torch.utils.data.DataLoader],
        batches_left: List[int],
        criterions: List[torch.nn.Module],
        force_task: Union[None, int] = None
) -> (torch.Tensor, torch.nn.Module, str, int):

    if len(dataloaders) != len(criterions):
        raise AttributeError('Cannot sample: number of dataloaders is not the '
                             'same as the number of criterions provided.')

    if force_task is None:
        probs = [x > 0 for x in batches_left]
        scaled_probs = [x / sum(probs) for x in probs]
        number_chosen = np.random.choice(range(len(batches_left)), p=scaled_probs)
    else:
        number_chosen = force_task

    batch = next(dataloaders[number_chosen])
    criterion = criterions[number_chosen]
    task = dataloaders[number_chosen]._dataset.task
    batches_left[number_chosen] -= 1

    if batches_left[number_chosen] == 0:
        logger.debug(f'Dataloader number {task} is exhausted.')

    return batch, criterion, task, number_chosen


def train_exhaust(
        model: nn.Module,
        train_dataloaders: List[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        criterions: List[torch.nn.Module],
        device: torch.device,
        verbose: bool = True,
        current_epoch: int = None,
        weights: List[int] = [1, 1, 1]
):
    model.train()

    if verbose:
        total_len = sum([len(x.dataset) for x in train_dataloaders])
        pbar = tqdm(total=total_len, leave=False,
                    desc=f'Training epoch {"" if current_epoch is None else current_epoch} on all tasks')


    iterable_dataloaders = [iter(x) for x in train_dataloaders]
    batches_left = [len(x) for x in train_dataloaders]
    not_exhausted_criterions = [x for x in criterions]

    while sum(batches_left) > 0:
        optimizer.zero_grad()

        batch, criterion, task, number_chosen = sample_task_from_pool(
            iterable_dataloaders,
            batches_left,
            not_exhausted_criterions
        )

        predictions = _batch_forward(batch, model, task, device)

        targets = batch['targets'].to(device)
        loss = criterion(predictions, targets).sum()
        loss.backward()

        if verbose:
            pbar.update(len(batch['targets']))

        for _ in range(weights[number_chosen] - 1):
            if batches_left[number_chosen] == 0:
                break

            batch, criterion, task, _ = sample_task_from_pool(
                iterable_dataloaders,
                batches_left,
                not_exhausted_criterions,
                number_chosen
            )

            predictions = _batch_forward(batch, model, task, device)

            targets = batch['targets'].to(device)
            loss = criterion(predictions, targets).sum()
            loss.backward()

            if verbose:
                pbar.update(len(batch['targets']))

        if weights[number_chosen] - 1 > 0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad /= weights[number_chosen]

        optimizer.step()

    if verbose:
        pbar.close()
