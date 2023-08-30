import numpy as np
from typing import Callable, Union, List

import pandas as pd
from tqdm import tqdm
import wandb

import torch
from torch import nn

from src.utils import logger, save_state


def _batch_forward(
        batch,
        model,
        task,
        device,
):
    if task == 'sentiment':
        ids, attention_masks = \
            batch['token_ids'], batch['attention_masks']

        ids = ids.to(device)
        attention_masks = attention_masks.to(device)

        predictions = model(task, ids, attention_masks)

    elif task == 'paraphrase_classifier' or task == 'paraphrase_regressor':
        ids_1, attention_masks_1, ids_2, attention_masks_2 = \
            (batch['token_ids_1'], batch['attention_masks_1'],
             batch['token_ids_2'], batch['attention_masks_2'])

        ids_1 = ids_1.to(device)
        ids_2 = ids_2.to(device)
        attention_masks_1 = attention_masks_1.to(device)
        attention_masks_2 = attention_masks_2.to(device)

        predictions = model(task, ids_1, attention_masks_1, ids_2, attention_masks_2)

    else:
        raise NotImplementedError

    return predictions


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
        logger.debug(f'Dataloader number {number_chosen} is exhausted.')

    return batch, criterion, task, number_chosen


def train_one_epoch_multitask(
        model: nn.Module,
        train_dataloaders: List[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        criterions: List[torch.nn.Module],
        device: torch.device,
        verbose: bool = True,
        current_epoch: int = None,
        weights: List[int] = [1, 1, 1]
):
    model.train()

    if verbose:
        total_len = sum([len(x.dataset) for x in train_dataloaders])
        if current_epoch is not None:
            pbar = tqdm(total=total_len, leave=False,
                        desc=f'Training epoch {current_epoch} on all tasks')
        else:
            pbar = tqdm(total=total_len, leave=False,
                        desc=f'Training model on all tasks')

    iterable_dataloaders = [iter(x) for x in train_dataloaders]
    batches_left = [len(x) for x in train_dataloaders]
    not_exhausted_criterions = [x for x in criterions]

    while sum(batches_left) > 0:
        optimizer.zero_grad()

        batch, criterion, task, number_chosen = sample_task_from_pool(
            iterable_dataloaders,
            batches_left,
            not_exhausted_criterions
        )

        predictions = _batch_forward(batch, model, task, device)

        targets = batch['targets'].to(device)
        loss = criterion(predictions, targets).sum()
        loss.backward()

        if verbose:
            pbar.update(len(batch['targets']))

        for _ in range(weights[number_chosen] - 1):
            if batches_left[number_chosen] == 0:
                break

            batch, criterion, task, _ = sample_task_from_pool(
                iterable_dataloaders,
                batches_left,
                not_exhausted_criterions,
                number_chosen
            )

            predictions = _batch_forward(batch, model, task, device)

            targets = batch['targets'].to(device)
            loss = criterion(predictions, targets).sum()
            loss.backward()

            if verbose:
                pbar.update(len(batch['targets']))

        if weights[number_chosen] - 1 > 0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad /= weights[number_chosen]

        optimizer.step()

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
        running_metric = torch.tensor(0, dtype=torch.float32).to(device)
        task = dataloader.dataset.task

        for batch in tqdm(dataloader, leave=False,
                          desc=f'Evaluating on {task}'):
            if task == 'sentiment':
                ids, attention_masks, targets = \
                    batch['token_ids'], batch['attention_masks'], batch['targets']

                ids = ids.to(device)
                attention_masks = attention_masks.to(device)
                targets = targets.to(device)

                predictions = model(task, ids, attention_masks)

            elif task == 'paraphrase_classifier' or task == 'paraphrase_regressor':
                ids_1, attention_masks_1, ids_2, attention_masks_2, targets = \
                    (batch['token_ids_1'], batch['attention_masks_1'],
                     batch['token_ids_2'], batch['attention_masks_2'],
                     batch['targets'])

                ids_1 = ids_1.to(device)
                ids_2 = ids_2.to(device)
                attention_masks_1 = attention_masks_1.to(device)
                attention_masks_2 = attention_masks_2.to(device)
                targets = targets.to(device)

                predictions = model(task, ids_1, attention_masks_1, ids_2, attention_masks_2)

            else:
                raise NotImplementedError

            if criterions is not None:
                loss = criterions[i](predictions, targets)
                running_loss += loss.item() * len(predictions)

            running_metric += metrics[i](predictions, targets) * len(predictions)

        if criterions:
            result[
                f'{task} {evaluation_set_name} loss'
            ] = running_loss / len(dataloader.dataset)

        result[
            f'{task} {evaluation_set_name} metric'
        ] = running_metric / len(dataloader.dataset)

    return result


def train_validation_loop_multitask(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: List[torch.nn.Module],
        metric: List[Callable[[torch.Tensor, torch.Tensor], float]],
        train_loader: List[torch.utils.data.DataLoader],
        val_loader: List[torch.utils.data.DataLoader],
        n_epochs: int,
        device: torch.device,
        weights: List[int] = [1, 1, 1],
        watcher: Union[str, None] = None,
        verbose: bool = True,
        save_best_path: str = None,
        overall_config: dict = None,
) -> dict:
    """
    Run the train loop with selected parameters while validating the model
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
            current_epoch=current_epoch,
            weights=weights
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
                         for key, value in epoch_train_scores.items()]
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
            evaluation_set_name='val'
        )

        score_strings = [f'{key}: {value:.3f}'
                         for key, value in epoch_val_scores.items()]
        score_message = ', '.join(score_strings)

        logger.info(f'Finished validating epoch {current_epoch}, '
                    + score_message)

        if result is None:
            result = {}
            for key, value in {**epoch_train_scores, **epoch_val_scores}.items():
                result[key] = [value, ]
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
        val_metric = (epoch_val_scores['sentiment val metric']
                      + epoch_val_scores['paraphrase_classifier val metric']
                      + (epoch_val_scores['paraphrase_regressor val metric'] + 1) / 2)

        if save_best_path is not None and val_metric > best_metric:
            save_state(model, optimizer, overall_config, save_best_path)

        current_epoch += 1

    logger.info(f'Finished training and validation the model.')

    return result


def train_loop_multitask(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: List[torch.nn.Module],
        train_loader: List[torch.utils.data.DataLoader],
        n_epochs: int,
        device: torch.device,
        verbose: bool = True
):
    """
    Run the train loop with selected parameters.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    criterion : torch.nn.Module
        The criterion (Loss function) to use (for example, MSELoss).
    train_loader : torch.utils.data.DataLoader
        The train DataLoader.
    n_epochs : int
        The number of epochs to train.
    device : torch.device
        The device to use while training.
    verbose : bool
        Whether the print statements are to be provided.
    """

    # Progress bar handling
    if verbose:
        pbar = tqdm(range(n_epochs))
    else:
        pbar = range(n_epochs)

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

        logger.info(f'Finished training epoch {current_epoch}')

        current_epoch += 1

    logger.info(f'Finished training and validation the model.')
