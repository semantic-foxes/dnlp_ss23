import random
from typing import Callable, Union, List

from tqdm import tqdm
import wandb

import torch
from torch import nn
from src.core.evaluation_multitask import evaluate_model_multitask, sum_comparator

from src.utils import logger, save_state


def sample_task_from_pool(
        dataloaders: List[torch.utils.data.DataLoader],
        batches_left: List[int],
        criterions: List[torch.nn.Module]
) -> (int, torch.utils.data.DataLoader, torch.nn.Module):

    if len(dataloaders) != len(criterions):
        raise AttributeError('Cannot sample: number of dataloaders is not the '
                             'same as the number of criterions provided.')

    number_chosen = random.choice(range(len(dataloaders)))
    batch = next(dataloaders[number_chosen])
    criterion = criterions[number_chosen]
    task = dataloaders[number_chosen]._dataset.task
    batches_left[number_chosen] -= 1

    if batches_left[number_chosen] == 0:
        logger.debug(f'Removing dataloader {task} since it is exhausted.')
        dataloaders.__delitem__(number_chosen)
        criterions.__delitem__(number_chosen)
        batches_left.__delitem__(number_chosen)

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
        total_len = sum([len(x.dataset) for x in train_dataloaders])
        if current_epoch is not None:
            pbar = tqdm(total=total_len, leave=False,
                        desc=f'Training epoch {current_epoch} on all tasks')
        else:
            pbar = tqdm(total=total_len, leave=False,
                        desc=f'Training model on all tasks')

    not_exhausted_dataloaders = [iter(x) for x in train_dataloaders]
    batches_left = [len(x) for x in train_dataloaders]
    not_exhausted_criterions = [x for x in criterions]

    while len(not_exhausted_dataloaders) > 0:
        batch, criterion, task = sample_task_from_pool(
            not_exhausted_dataloaders,
            batches_left,
            not_exhausted_criterions
        )
        optimizer.zero_grad()
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

        loss = criterion(predictions, targets).sum()
        loss.backward()
        optimizer.step()

        if verbose:
            pbar.update(len(batch['targets']))

    if verbose:
        pbar.close()




def train_validation_loop_multitask(
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
        overall_config: dict = None,
        metric_comparator: Callable[[dict, dict], bool] = sum_comparator
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
    metric_comparator: Callable[[dict, dict], bool]
        Compares evaluated metrics 

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

    best_metric = {'sentiment': 0, 'paraphrase_classifier': 0, 'paraphrase_regressor': -1}
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

        logger.info(f'Training results for epoch {current_epoch}')
        epoch_train_scores = evaluate_model_multitask(
            model,
            train_loader,
            device,
            metric,
            criterion
        )

        # Validation
        logger.info(f'Validation results for epoch {current_epoch}')
        epoch_val_scores = evaluate_model_multitask(
            model,
            val_loader,
            device,
            metric,
            criterion
        )


        if result is None:
            result = {}

        result[current_epoch] = {'train': epoch_train_scores, 'val': epoch_val_scores}

        # Upload to watcher
        if watcher is not None:
            try:
                watcher_command({**epoch_train_scores, **epoch_val_scores})
            except Exception as e:
                logger.error(f'Error loading to watcher at epoch {current_epoch}')
                raise e

        if save_best_path is not None and metric_comparator(epoch_val_scores['metric'], best_metric):
            best_metric = epoch_val_scores['metric']
            save_state(model, optimizer, overall_config, save_best_path)

        current_epoch += 1

    logger.info(f'Finished training and validation the model.')

    return result
