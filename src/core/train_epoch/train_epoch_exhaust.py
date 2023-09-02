from typing import List

from tqdm import tqdm

import torch
from torch import nn

from src.core.train_epoch.train_batch import train_one_batch_multitask
from src.core.train_epoch.sample_task import sample_task_from_pool


def train_epoch_exhaust(
        model: nn.Module,
        train_dataloaders: List[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        criterions: List[torch.nn.Module],
        device: torch.device,
        train_mode: str,
        overall_config: dict = None,
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

    def pbar_update(pbar, batch, task):
        if train_mode == 'standard' or task == 'sentiment':
            batch_size = len(batch['targets'])
        elif train_mode == 'contrastive':
            batch_size = len(batch[0]['targets'])
        elif train_mode == 'triplet':
            batch_size = len(batch[list(batch.keys())[0]])
        pbar.update(batch_size)

    while sum(batches_left) > 0:
        optimizer.zero_grad()

        batch, criterion, task, number_chosen = sample_task_from_pool(
            iterable_dataloaders,
            batches_left,
            not_exhausted_criterions
        )

        train_one_batch_multitask(model, batch, optimizer, criterion, device, task, train_mode, overall_config)

        if verbose:
            pbar_update(pbar, batch, task)

        for _ in range(weights[number_chosen] - 1):
            if batches_left[number_chosen] == 0:
                break

            batch, criterion, task, _ = sample_task_from_pool(
                iterable_dataloaders,
                batches_left,
                not_exhausted_criterions,
                number_chosen
            )

            train_one_batch_multitask(model, batch, optimizer, criterion, device, task, train_mode, overall_config)

            if verbose:
                pbar_update(pbar, batch, task)

        if weights[number_chosen] - 1 > 0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad /= weights[number_chosen]

        optimizer.step()

    if verbose:
        pbar.close()
