import random
from typing import Callable, Union, List

from tqdm import tqdm
import wandb

import torch
from torch import nn
from src.core.evaluation_multitask import evaluate_model_multitask, sum_comparator
from src.core.train_epoch.train_epoch_continuos import train_continuos
from src.core.train_epoch.train_epoch_exhaust import train_exhaust
from src.core.train_epoch.train_epoch_min import train_min
from src.core.train_epoch.train_epoch_sequential import train_sequential


from src.utils import logger, save_state
from src.utils.model_utils import save_results

def train_one_epoch_multitask(
        model: nn.Module,
        train_dataloaders: List[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        criterions: List[torch.nn.Module],
        device: torch.device,
        data_combine: str = 'sequential',
        verbose: bool = True,
        current_epoch: int = None,
        prev_state = None
):
    model.train()
    if data_combine == 'exhaust':
        train_exhaust(model, train_dataloaders, optimizer, criterions, device, verbose, current_epoch)
        return 
    
    if data_combine == 'sequential':
        train_sequential(model, train_dataloaders, optimizer, criterions, device, verbose, current_epoch)
        return 
    
    if data_combine == 'min':
        train_min(model, train_dataloaders, optimizer, criterions, device, verbose, current_epoch)
        return 

    if data_combine == 'continuos':
        return train_continuos(
            model, train_dataloaders, optimizer, criterions, device, verbose, current_epoch, 
            prev_data_iters=prev_state
        )
         

    message = f'{data_combine} is not known data combine strategy.'
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
        watcher: Union[str, None] = None,
        verbose: bool = True,
        save_best_path: str = None,
        overall_config: dict = None,
        metric_comparator: Callable[[dict, dict], bool] = sum_comparator,
        data_combine: str = 'sequential',
        skip_train_eval: int = 1,
        best_metric: dict = {},
        result: List = [],
        results_path: str = 'results/results.csv',
):
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
    data_combine: 
        strategy to combine training datasets 'exhaust' or 'sequential' or 'min' or 'continuos'
    skip_train_eval: 
        skip every n-th evaluation of training datasets
    best_metric: 
        stores best metric results from previous training
    result: List
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
    best_metric = {'sentiment': 0, 'paraphrase_classifier': 0, 'paraphrase_regressor': -1, **best_metric}
    current_epoch = 0

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
            verbose=True,
            current_epoch=current_epoch,
            data_combine=data_combine,
            prev_state=epoch_train_state
        )

        logger.info(f'Finished training epoch {current_epoch}')
        if current_epoch % skip_train_eval == 0:
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

        result.append({'train': epoch_train_scores, 'val': epoch_val_scores})

        # Upload to watcher
        if watcher is not None:
            try:
                watcher_command({**epoch_train_scores, **epoch_val_scores})
            except Exception as e:
                logger.error(f'Error loading to watcher at epoch {current_epoch}')
                raise e
            
        # in case of partial evaluation i.e only on sst dataset
        current_metric = {**best_metric, **epoch_val_scores['metric']}
        if save_best_path is not None and metric_comparator(current_metric, best_metric):
            best_metric = current_metric
            save_state(model, optimizer, overall_config, save_best_path)

        current_epoch += 1

    logger.info(f'Finished training and validation the model.')
    save_results(result, results_path)

    return result, best_metric
