from typing import Union, List

import numpy as np

import torch

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
        logger.debug(f'Dataloader for {task} is exhausted.')

    return batch, criterion, task, number_chosen
