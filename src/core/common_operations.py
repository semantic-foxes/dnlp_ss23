from typing import Callable, Union, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

import torch
from torch import nn

from src.utils import logger, save_state


@torch.no_grad()
def evaluate_model(
        model: nn.Module,
        eval_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        metric: Callable,
        criterion: torch.nn.Module = None,
        dataloader_message: str = 'train'
) -> Union[float, List[float]]:
    """
    Evaluates the model using the given dataloader

    Parameters
    ----------
    model : nn.Module,
    eval_dataloader : torch.utils.data.Dataloader
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
    running_loss = 0
    running_metric = 0

    for batch in tqdm(eval_dataloader,
                      desc=f'Evaluating on {dataloader_message}', leave=False):
        ids, mask, targets, = \
            batch['token_ids'], batch['attention_masks'], batch['targets']

        ids = ids.to(device)
        mask = mask.to(device)
        targets = targets.to(device)

        logits = model(ids, mask)

        if criterion is not None:
            loss = criterion(logits, targets)
            running_loss += loss.item() * len(logits)

        predictions = np.argmax(logits.detach().cpu().numpy(), axis=1).flatten()
        running_metric += metric(predictions, targets.cpu().numpy()) * len(predictions)

    if criterion:
        return running_loss / len(eval_dataloader.dataset), \
            running_metric / len(eval_dataloader.dataset)
    else:
        return running_metric / len(eval_dataloader.dataset)


@torch.no_grad()
def generate_predictions(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        dataloader_message: str = 'test'
) -> pd.DataFrame:
    """
    Generates predictions for the given dataloader.

    Parameters
    ----------
    model : nn.Module
    dataloader : torch.utils.data.Dataloader
    device : torch.device
    dataloader_message : str
        The message to be put in the `tqdm` progress bar.

    Returns
    -------
    result: pd.DataFrame
        The resulting predictions.
    """

    model.eval()
    # result = pd.DataFrame(columns=['prediction'])
    task = dataloader.dataset.task
    progress_bar = tqdm(dataloader,
                        desc=f'Making predictions on {dataloader_message}')

    result = []

    for i, batch in enumerate(progress_bar):
        if task == 'sentiment':
            ids, attention_masks = \
                batch['token_ids'], batch['attention_masks']

            ids = ids.to(device)
            attention_masks = attention_masks.to(device)

            predictions = model(task, ids, attention_masks)
            predictions = torch.argmax(predictions, dim=1)

        elif task == 'paraphrase_classifier':
            ids_1, attention_masks_1, ids_2, attention_masks_2 = \
                (batch['token_ids_1'], batch['attention_masks_1'],
                 batch['token_ids_2'], batch['attention_masks_2'])

            ids_1 = ids_1.to(device)
            ids_2 = ids_2.to(device)
            attention_masks_1 = attention_masks_1.to(device)
            attention_masks_2 = attention_masks_2.to(device)

            predictions = model(task, ids_1, attention_masks_1, ids_2, attention_masks_2)
            predictions = torch.argmax(predictions, dim=1)

        elif task == 'paraphrase_regressor':
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
        result += predictions.cpu().numpy().tolist()

    result = pd.DataFrame(
        {'prediction': result},
        index=dataloader.dataset.ids
    )

    return result


def train_one_epoch(
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        device: torch.device,
        verbose: bool = True,
        current_epoch: int = None,
):
    model.train()

    if verbose:
        if current_epoch is not None:
            pbar = tqdm(train_dataloader, leave=False,
                        desc=f'Training epoch {current_epoch}')
        else:
            pbar = tqdm(train_dataloader, leave=False,
                        desc=f'Training model')
    else:
        pbar = train_dataloader

    for batch in pbar:
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


def train_validation_loop(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        metric: Callable[[torch.Tensor, torch.Tensor], float],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        n_epochs: int,
        device: torch.device,
        watcher: Union[str, None] = None,
        verbose: bool = True,
        save_best_path: str = None,
        overall_config: dict = None
) -> Tuple[list, list, list, list]:
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
    train_loss_array = []
    val_loss_array = []
    train_metric_array = []
    val_metric_array = []

    best_metric = 0
    current_epoch = 0

    logger.info('Starting training and validating the model.')
    for _ in pbar:
        # Train
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            verbose=True,
            current_epoch=current_epoch
        )

        train_loss, train_metric = evaluate_model(
            model,
            train_loader,
            device,
            metric,
            criterion,
            dataloader_message='train'
        )

        train_loss_array.append(train_loss)
        train_metric_array.append(train_metric)

        if watcher is not None:
            watcher_dict = {
                'Train loss': train_loss,
                'Train metric': train_metric
            }
        logger.info(f'Finished training epoch {current_epoch}, '
                    f'train loss: {train_loss:.3f}, train metric: {train_metric:.3f}.')

        # Validation
        val_loss, val_metric = evaluate_model(
            model,
            val_loader,
            device,
            metric,
            criterion,
            dataloader_message='val'
        )

        val_loss_array.append(val_loss)
        val_metric_array.append(val_metric)

        if watcher is not None:
            watcher_dict = {
                **watcher_dict,
                'Val loss': val_loss,
                'Val metric': val_metric
            }
        logger.info(f'Finished validating epoch {current_epoch}, '
                    f'val loss: {val_loss:.3f}, val metric: {val_metric:.3f}.')

        # Upload to watcher
        if watcher is not None:
            try:
                watcher_command(watcher_dict)
            except Exception as e:
                logger.error(f'Error loading to watcher at epoch {current_epoch}')
                raise e

        if save_best_path is not None and val_metric > best_metric:
            best_metric = val_metric
            save_state(model, optimizer, overall_config, save_best_path)

        current_epoch += 1

    logger.info(f"Finished training and validation the model. Best val metric: {best_metric:.3f}")

    return train_loss_array, val_loss_array, train_metric_array, val_metric_array



