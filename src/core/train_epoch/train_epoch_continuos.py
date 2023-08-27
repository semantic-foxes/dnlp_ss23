import random
from typing import List

from tqdm import tqdm

import torch
from torch import nn

from src.core.train_epoch.train_batch import train_one_batch_multitask

def train_continuos(
        model: nn.Module,
        train_dataloaders: List[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        criterions: List[torch.nn.Module],
        device: torch.device,
        verbose: bool = True,
        current_epoch: int = None,
        prev_data_iters: List[iter] = None
):
    model.train()

    min_batches = min([len(x) for x in train_dataloaders])
    if verbose:
        pbar = tqdm(range(min_batches), leave=False,
                    desc=f'Training epoch {"" if current_epoch is None else current_epoch} on all tasks')
        
    data_iters = [iter(x) for x in train_dataloaders] if prev_data_iters is None else prev_data_iters

    for _ in pbar:  
        for i, (data_iter, criterion) in enumerate(zip(data_iters, criterions)):
            task = data_iter._dataset.task
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iters[i] = iter(train_dataloaders[i])
                batch = next(data_iters[i])
            train_one_batch_multitask(model, batch, optimizer, criterion, device, task)
    return data_iters
