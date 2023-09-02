from typing import List

from tqdm import tqdm

import torch
from torch import nn

from src.core.train_epoch.train_batch import _batch_forward
from src.core import sample_task_from_pool


def train_epoch_exhaust(
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
        total_len = sum([len(x) for x in train_dataloaders])
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
            pbar.update(1)

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
