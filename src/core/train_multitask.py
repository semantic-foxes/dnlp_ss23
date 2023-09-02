from typing import Callable, Union, List, Iterable

from tqdm import tqdm
import wandb

import torch
from torch import nn
from src.core.evaluation_multitask import evaluate_model_multitask, sum_comparator
from src.core.train_epoch.train_epoch_continuous import train_epoch_continuous
from src.core.train_epoch.train_epoch_exhaust import train_epoch_exhaust
from src.core.train_epoch.train_epoch_sequential import train_epoch_sequential
from src.utils import logger, save_state


def train_one_epoch_multitask(
        model: nn.Module,
        train_dataloaders: List[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        criterions: List[torch.nn.Module],
        device: torch.device,
        dataloader_mode: str = 'sequential',
        verbose: bool = True,
        current_epoch: int = None,
        prev_state: List[Iterable] = None,
        weights: List[int] = [1, 10, 1],
        skip_optimizer_step: int = 1,
        cosine_loss = None,
        overall_config = {},
):
    if dataloader_mode == 'exhaust':
        train_epoch_exhaust(
            model=model,
            train_dataloaders=train_dataloaders,
            optimizer=optimizer,
            criterions=criterions,
            device=device,
            verbose=verbose,
            current_epoch=current_epoch,
            weights=weights
        )
    
    elif dataloader_mode == 'sequential':
        train_epoch_sequential(
            model=model,
            train_dataloaders=train_dataloaders,
            optimizer=optimizer,
            criterions=criterions,
            device=device,
            verbose=verbose,
            current_epoch=current_epoch,
        )
    
    elif dataloader_mode == 'min':
        train_epoch_sequential(
            model=model,
            train_dataloaders=train_dataloaders,
            optimizer=optimizer,
            criterions=criterions,
            device=device,
            verbose=verbose,
            current_epoch=current_epoch,
            cut_to_min_size=True,
            skip_optimizer_step=skip_optimizer_step,
        )

    elif dataloader_mode == 'continuous':
        # if skip_optimizer_step > 1:
        #     skip_optimizer_step += 2 * current_epoch
        train_epoch_continuous(
            model=model,
            train_dataloaders=train_dataloaders,
            optimizer=optimizer,
            criterions=criterions,
            device=device,
            verbose=verbose,
            current_epoch=current_epoch,
            prev_data_iters=prev_state,
            skip_optimizer_step=skip_optimizer_step,
            cosine_loss=cosine_loss,
            weights=weights,
            overall_config=overall_config,
        )

    else:
        message = f'{dataloader_mode} is not a known data combine strategy.'
        logger.error(message)
        raise NotImplementedError(message)
    

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
        metric_comparator: Callable[[dict, dict], bool] = sum_comparator,
        dataloader_mode: str = 'sequential',
        skip_train_eval: int = 1,
        best_metric: dict = {},
        prior_scores: List = [],
        skip_optimizer_step: int = 1,
        cosine_loss = None,
):
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
    metric_comparator: Callable[[dict, dict], bool]
        Compares evaluated metrics 
    dataloader_mode: 
        strategy to combine training datasets 'exhaust' or 'sequential' or 'min' or 'continuous'
    skip_train_eval: 
        skip every n-th evaluation of training datasets
    best_metric: 
        stores best metric results from previous training
    prior_scores: List
        stores all previous scores

    Returns
    -------
    result: List
        The resulting list, train loss, train metric, validation loss
        and validation metric lists.
    best_metric:
        best metric results after training
    """
    # Save handling
    if save_best_path and overall_config is None:
        message = 'No model config is provided while save path is provided.'
        logger.error(message)
        raise AttributeError(message)

    # Progress bar handling
    pbar = range(n_epochs)

    # Watcher handling
    if watcher == 'wandb':
        watcher_command = wandb.log
    elif watcher is not None:
        message = 'Watchers except WandB are not implemented yet.'
        logger.error(message)
        raise NotImplementedError(message)

    # Initialization
    best_metric = {
        'sentiment': 0,
        'paraphrase_classifier': 0,
        'paraphrase_regressor': -1,
        **best_metric
    }
    current_epoch = 0
    resulting_scores = []

    logger.info('Starting training and validating the model.')
    epoch_train_state = None
    for _ in pbar:
        # Train
        epoch_train_state = train_one_epoch_multitask(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            verbose=verbose,
            current_epoch=current_epoch,
            weights=weights,
            dataloader_mode=dataloader_mode,
            prev_state=epoch_train_state,
            skip_optimizer_step=skip_optimizer_step,
            cosine_loss=cosine_loss,
            overall_config=overall_config,
        )
        logger.info(f'Finished training epoch {current_epoch}')

        current_epoch_scores = {}
        # Validation on train
        if current_epoch % skip_train_eval == 0:
            logger.info(f'Training results for epoch {current_epoch}')
            epoch_train_scores = evaluate_model_multitask(
                model,
                train_loader,
                device,
                metric,
                criterion,
                cosine_loss,
                overall_config,
                verbose,
            )
            current_epoch_scores['train'] = epoch_train_scores

        # Validation on val
        logger.info(f'Validation results for epoch {current_epoch}')
        epoch_val_scores = evaluate_model_multitask(
            model,
            val_loader,
            device,
            metric,
            criterion,
            cosine_loss,
            overall_config,
            verbose,
        )
        current_epoch_scores['val'] = epoch_val_scores

        resulting_scores = [*prior_scores, current_epoch_scores]

        # Upload to watcher
        if watcher is not None:
            try:
                watcher_command(current_epoch_scores)
            except Exception as e:
                logger.error(f'Error loading to watcher at epoch {current_epoch}')
                raise e

        # In case we now train on smaller number of datasets than when
        # we obtained the best metric.
        current_metric = {**best_metric, **epoch_val_scores['metric']}
        if save_best_path is not None and metric_comparator(current_metric, best_metric):
            best_metric = current_metric
            save_state(model, optimizer, overall_config, save_best_path)

        current_epoch += 1

    logger.info(f'Finished training and validation the model.')

    return resulting_scores, best_metric
