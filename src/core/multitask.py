import random
from typing import Callable, Union, List

import numpy as np
from tqdm import tqdm
import wandb

import torch
from torch import nn

from src.utils import logger, save_state


def sample_task_from_pool(
        dataloaders: List[torch.utils.data.DataLoader],
        criterions: List[torch.nn.Module]
) -> (int, torch.utils.data.DataLoader, torch.nn.Module):

    if len(dataloaders) != len(criterions):
        raise AttributeError('Cannot sample: number of dataloaders is not the '
                             'same as the number of criterions provided.')

    number_chosen = random.choice(range(len(dataloaders)))

    try:
        batch = next(dataloaders[number_chosen])
        criterion = criterions[number_chosen]
        task = dataloaders[number_chosen].dataset.task
    except StopIteration:
        dataloaders.__delitem__(number_chosen)
        criterions.__delitem__(number_chosen)
        batch, criterion, task = sample_task_from_pool(dataloaders, criterions)

    return batch, criterion, task


def train_one_epoch_multitask(
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
        total_len = sum([len(x) for x in train_dataloaders])
        if current_epoch is not None:
            pbar = tqdm(total=total_len, leave=False,
                        desc=f'Training epoch {current_epoch} on all tasks')
        else:
            pbar = tqdm(total=total_len, leave=False,
                        desc=f'Training model on all tasks')

    not_exhausted_dataloaders = [x for x in train_dataloaders]
    not_exhausted_criterions = [x for x in criterions]

    while len(not_exhausted_dataloaders) > 0:
        batch, criterion, task = sample_task_from_pool(not_exhausted_dataloaders,
                                                       not_exhausted_criterions)

        ids, attention_masks, targets = \
            batch['token_ids'], batch['attention_masks'], batch['targets']

        ids = ids.to(device)
        attention_masks = attention_masks.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(ids, attention_masks)
        loss = criterion(logits, targets.reshape(-1)).sum()
        loss.backward()
        optimizer.step()

        if verbose:
            pbar.update(len(ids))

    if verbose:
        pbar.close()


@torch.no_grad()
def evaluate_model_multitask(
        model: nn.Module,
        eval_dataloaders: List[torch.utils.data.DataLoader],
        device: torch.device,
        metrics: List[Callable],
        criterions: List[torch.nn.Module] = None,
        evaluation_set_name: str = 'train'
) -> dict:
    """
    Evaluates the model using the given dataloader

    Parameters
    ----------
    model : nn.Module,
    eval_dataloaders : torch.utils.data.Dataloader
    device : torch.device
    metric : Callable
        The metric function. It is expected to have the signature of
        (y_true, y_predicted).
    criterion : torch.nn.Module
        The criterion (loss function) to use.
    dataloader_message : str
        The message to be put in the `tqdm` progress bar.

    Returns
    -------
    result: [float, float] or float
        The resulting criterion and metric or just the metric if no criterion
        is provided.
    """

    model.eval()
    if type(eval_dataloaders) is not list and type(eval_dataloaders) is not tuple:
        eval_dataloaders = [eval_dataloaders, ]

    result = {}

    for i, dataloader in enumerate(eval_dataloaders):
        running_loss = 0
        running_metric = 0
        task_name = dataloader.dataset.task

        for batch in tqdm(dataloader, leave=False,
                          desc=f'Evaluating on {task_name}'):
            ids, mask, targets, = \
                batch['token_ids'], batch['attention_masks'], batch['targets']

            ids = ids.to(device)
            mask = mask.to(device)
            targets = targets.to(device)

            logits = model(ids, mask)

            if criterions is not None:
                loss = criterions[i](logits, targets)
                running_loss += loss.item() * len(logits)

            predictions = np.argmax(logits.detach().cpu().numpy(), axis=1).flatten()
            running_metric += metrics[i](predictions, targets.cpu().numpy()) * len(predictions)

        if criterions:
            result[
                f'{task_name} {evaluation_set_name} loss'
            ] = running_loss / len(dataloader.dataset)

        result[
            f'{task_name} {evaluation_set_name} metric'
        ] = running_metric / len(dataloader.dataset)

    return result


def train_validation_loop(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: List[torch.nn.Module],
        metric: List[Callable[[torch.Tensor, torch.Tensor], float]],
        train_loader: List[torch.utils.data.DataLoader],
        val_loader: List[torch.utils.data.DataLoader],
        n_epochs: int,
        device: torch.device,
        watcher: Union[str, None] = None,
        verbose: bool = True,
        save_best_path: str = None,
        overall_config: dict = None
) -> dict:
    """
    Run the train loop with selecting parameters while validating the model
    after each epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    criterion : torch.nn.Module
        The criterion (Loss function) to use (for example, MSELoss).
    metric : Callable[[torch.Tensor, torch.Tensor], float]
        The metric to calculate during training.
    train_loader : torch.utils.data.DataLoader
        The train DataLoader.
    val_loader : torch.utils.data.DataLoader
        The validation DataLoader.
    n_epochs : int
        The number of epochs to train.
    device : torch.device
        The device to use while training.
    watcher : Union[str, None]
        The watcher to use.
    verbose : bool
        Whether the print statements are to be provided.
    save_best_path : str
        The path to save the best model. If not provided, the best model will
        not be saved. The best model is decided based on the val metric score.
    overall_config : dict
        The overall config (should include all the data about the model and
        the datasets, etc.). Required if the save_path is provided.

    Returns
    -------
    result: tuple of lists
        The resulting list, train loss, train metric, validation loss
        and validation metric lists.
    """
    # Save handling
    if save_best_path and overall_config is None:
        message = 'No model config is provided while save path is provided.'
        logger.error(message)
        raise AttributeError(message)

    # Progress bar handling
    if verbose:
        pbar = tqdm(range(n_epochs))
    else:
        pbar = range(n_epochs)

    # Watcher handling
    if watcher == 'wandb':
        watcher_command = wandb.log
    elif watcher is not None:
        message = 'Watchers except WandB are not implemented yet.'
        logger.error(message)
        raise NotImplementedError(message)

    # Initialization
    result = None

    best_metric = 0
    current_epoch = 0

    logger.info('Starting training and validating the model.')
    for _ in pbar:
        # Train
        train_one_epoch_multitask(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            verbose=True,
            current_epoch=current_epoch
        )

        epoch_train_scores = evaluate_model_multitask(
            model,
            train_loader,
            device,
            metric,
            criterion,
            evaluation_set_name='train'
        )

        score_strings = [f'{key}: {value:.3f}'
                         for key, value in epoch_train_scores]
        score_message = ', '.join(score_strings)

        logger.info(f'Finished training epoch {current_epoch}, '
                    + score_message)

        # Validation
        epoch_val_scores = evaluate_model_multitask(
            model,
            val_loader,
            device,
            metric,
            criterion,
            dataloader_message='val'
        )

        score_strings = [f'{key}: {value:.3f}'
                         for key, value in epoch_val_scores]
        score_message = ', '.join(score_strings)

        logger.info(f'Finished validating epoch {current_epoch}, '
                    + score_message)

        if current_epoch == 0:
            result = {**epoch_train_scores, **epoch_val_scores}
        else:
            for key, value in {**epoch_train_scores, **epoch_val_scores}.items():
                result[key].append(value)

        # Upload to watcher
        if watcher is not None:
            try:
                watcher_command({**epoch_train_scores, **epoch_val_scores})
            except Exception as e:
                logger.error(f'Error loading to watcher at epoch {current_epoch}')
                raise e

        # TODO: What is the criterion for model saving?
        # if save_best_path is not None and val_metric > best_metric:
        #     save_state(model, optimizer, overall_config, save_best_path)

        current_epoch += 1

    logger.info(f'Finished training and validation the model.')

    return result
