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
        train_mode: str,
        overall_config: dict = None,
        cut_to_min_size: bool = False,
        verbose: bool = True,
        current_epoch: int = None,
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

    for dataloader, criterion in zip(train_dataloaders, criterions):
        task = dataloader.dataset.task
        batches_used = 0

        for batch in dataloader:
            train_one_batch_multitask(model, batch, optimizer, criterion, device, task, train_mode, overall_config)
            batches_used += 1

            if verbose:
                pbar.update(len(batch))

            if cut_to_min_size and batches_used >= cutoff:
                break


