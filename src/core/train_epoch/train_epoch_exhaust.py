import random
from typing import List

from tqdm import tqdm

import torch
from torch import nn

from src.core.train_epoch.train_batch import train_one_batch_multitask

from src.utils import logger


def sample_exhaust(
        dataloaders: List[torch.utils.data.DataLoader],
        batches_left: List[int],
        criterions: List[torch.nn.Module]
) -> (int, torch.utils.data.DataLoader, torch.nn.Module):

    if len(dataloaders) != len(criterions):
        raise AttributeError('Cannot sample: number of dataloaders is not the '
                             'same as the number of criterions provided.')

    number_chosen = random.choice(range(len(dataloaders)))
    batch = next(dataloaders[number_chosen])
    criterion = criterions[number_chosen]
    task = dataloaders[number_chosen]._dataset.task
    batches_left[number_chosen] -= 1

    if batches_left[number_chosen] == 0:
        logger.debug(f'Removing dataloader {task} since it is exhausted.')
        dataloaders.__delitem__(number_chosen)
        criterions.__delitem__(number_chosen)
        batches_left.__delitem__(number_chosen)

    return batch, criterion, task


def train_exhaust(
        model: nn.Module,
        train_dataloaders: List[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        criterions: List[torch.nn.Module],
        device: torch.device,
        verbose: bool = True,
        current_epoch: int = None,
):
    model.train()

    if verbose:
        total_len = sum([len(x.dataset) for x in train_dataloaders])
        pbar = tqdm(total=total_len, leave=False,
                    desc=f'Training epoch {current_epoch is None if "" else current_epoch} on all tasks')


    not_exhausted_dataloaders = [iter(x) for x in train_dataloaders]
    batches_left = [len(x) for x in train_dataloaders]
    not_exhausted_criterions = [x for x in criterions]

    while len(not_exhausted_dataloaders) > 0:
        batch, criterion, task = sample_exhaust(
            not_exhausted_dataloaders,
            batches_left,
            not_exhausted_criterions
        )
        train_one_batch_multitask(model, batch, optimizer, criterion, device, task)

        if verbose:
            pbar.update(len(batch['targets']))

    if verbose:
        pbar.close()

