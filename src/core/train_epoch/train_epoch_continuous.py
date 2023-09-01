from typing import List
import numpy as np

from tqdm import tqdm

import torch
from torch import nn

from src.core.train_epoch.train_batch import train_one_batch_multitask
from src.utils import logger


def train_epoch_continuous(
        model: nn.Module,
        train_dataloaders: List[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        criterions: List[torch.nn.Module],
        device: torch.device,
        number_of_batches: int = None,
        verbose: bool = True,
        current_epoch: int = None,
        prev_data_iters: List[iter] = None,
        skip_optimizer_step: int = 1,
        weights: List[int] = [1, 10, 1], 
        cosine_loss = None,
):
    """
    Train mode in which the dataloaders are restarted until the required
    number of batches is not reached. By default, the number of batches
    is equal to the size of the minimal dataloader times 3.

    Parameters
    ----------
    model
    train_dataloaders
    optimizer
    criterions
    device
    number_of_batches
    verbose
    current_epoch
    prev_data_iters

    Returns
    -------

    """
    model.train()

    if number_of_batches is None:
        number_of_batches = min([len(x) for x in train_dataloaders]) * len(train_dataloaders)

    if verbose:
        pbar = tqdm(range(number_of_batches), leave=False,
                    desc=f'Training epoch {"" if current_epoch is None else current_epoch} on all tasks')
    else:
        pbar = range(number_of_batches)

    data_iters = [iter(x) for x in train_dataloaders] if prev_data_iters is None else prev_data_iters

    optimizer.zero_grad()

    def sample_and_train_batch(number_chosen, is_optimizer_step=True, loss_divisor=1):
        task = data_iters[number_chosen]._dataset.task
        # reset a dataloader if ended
        try:
            batch = next(data_iters[number_chosen])
        except StopIteration:
            data_iters[number_chosen] = iter(train_dataloaders[number_chosen])
            logger.info(f'Resetting {task} dataloader.')

            batch = next(data_iters[number_chosen])

        train_one_batch_multitask(
            model,
            batch,
            optimizer,
            criterion=criterions[number_chosen],
            device=device,
            task=task,
            is_optimizer_step=is_optimizer_step,
            loss_divisor=loss_divisor,
            cosine_loss=cosine_loss,
        )

    for i in pbar:
        number_chosen = np.random.choice(range(len(data_iters)))
        weight = weights[number_chosen]

        is_optimizer_step = (1+i) % skip_optimizer_step == 0

        # squeeze extra data
        for _ in range(weight-1):
            sample_and_train_batch(number_chosen, False, 1/weight)
        
        sample_and_train_batch(number_chosen, is_optimizer_step, 1/weight)

    return data_iters
