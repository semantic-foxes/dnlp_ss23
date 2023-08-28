import random
from typing import List

from tqdm import tqdm

import torch
from torch import nn

from src.core.train_epoch.train_batch import train_one_batch_multitask

def train_sequential(
        model: nn.Module,
        train_dataloaders: List[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        criterions: List[torch.nn.Module],
        device: torch.device,
        verbose: bool = True,
        current_epoch: int = None,
):
    model.train()

    for dataloader, criterion in zip(train_dataloaders, criterions):
        task = dataloader.dataset.task
        if verbose:
            pbar = tqdm(dataloader, leave=False,
                        desc=f'Training epoch {"" if current_epoch is None else current_epoch} on {task}')
        for batch in pbar:
            train_one_batch_multitask(model, batch, optimizer, criterion, device, task)


