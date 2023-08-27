import random
from typing import List

from tqdm import tqdm

import torch
from torch import nn

from src.core.train_epoch.train_batch import train_one_batch_multitask

def train_min(
        model: nn.Module,
        train_dataloaders: List[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        criterions: List[torch.nn.Module],
        device: torch.device,
        verbose: bool = True,
        current_epoch: int = None,
):
    model.train()

    min_batches = min([len(x) for x in train_dataloaders])
    if verbose:
        pbar = tqdm(range(min_batches), leave=False,
                    desc=f'Training epoch {"" if current_epoch is None else current_epoch} on all tasks')
        
    data_iters = [iter(x) for x in train_dataloaders]
    for _ in pbar:  
        for data_iter, criterion in zip(data_iters, criterions):
            task = data_iter._dataset.task
            batch = next(data_iter)
            train_one_batch_multitask(model, batch, optimizer, criterion, device, task)
