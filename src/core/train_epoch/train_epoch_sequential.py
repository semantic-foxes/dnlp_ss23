from typing import List

from tqdm import tqdm

import torch
from torch import nn

from src.core.train_epoch.train_batch import train_one_batch_multitask


def train_epoch_sequential(
        model: nn.Module,
        train_dataloaders: List[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        criterions: List[torch.nn.Module],
        device: torch.device,
        cut_to_min_size: bool = False,
        verbose: bool = True,
        current_epoch: int = None,
        skip_optimizer_step: int = 1,
):
    """
    Train using each of the dataloaders consequently with no random involved.
    Cut_to_min_size is used mainly as a debug option.

    Parameters
    ----------
    model
    train_dataloaders
    optimizer
    criterions
    device
    cut_to_min_size
    verbose
    current_epoch

    Returns
    -------

    """
    model.train()

    if cut_to_min_size:
        cutoff = min([len(x) for x in train_dataloaders])

    if verbose:
        if cut_to_min_size:
            pbar = tqdm(
                range(cutoff * len(train_dataloaders)),
                leave=False,
                desc=f'Training epoch {"" if current_epoch is None else current_epoch} on all tasks'
            )
        else:
            pbar = tqdm(
                range(sum([len(x) for x in train_dataloaders])),
                leave=False,
                desc=f'Training epoch {"" if current_epoch is None else current_epoch} on all tasks'
            )

    optimizer.zero_grad()
    
    for dataloader, criterion in zip(train_dataloaders, criterions):
        task = dataloader.dataset.task
        batches_used = 0

        for batch in dataloader:
            batches_used += 1
            is_optimizer_step = batches_used % skip_optimizer_step == 0
            train_one_batch_multitask(
                model, batch, optimizer, criterion, device, task, 
                is_optimizer_step=is_optimizer_step,
                loss_divisor=skip_optimizer_step,
            )

            if verbose:
                pbar.update(1)

            if cut_to_min_size and batches_used >= cutoff:
                break

    if verbose:
        pbar.close()

