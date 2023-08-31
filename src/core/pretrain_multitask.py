from typing import Callable, Union, List

import torch
from src.core.evaluation_multitask import sum_comparator
from src.core.train_multitask import train_validation_loop_multitask
from src.utils.model_utils import load_state
from src.utils import logger


def pretrain_validation_loop_multitask(
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
):
    """
    Run the train loop with selecting parameters while validating the model
    after each epoch.
    """
    best_metric = {**best_metric}

    for i in range(len(metric)):
        try:
            load_state(model, device, save_best_path)
        except:
            logger.info(f'failed to load model from {save_best_path}')

        result, best_metric = train_validation_loop_multitask(
            model=model,
            optimizer=optimizer,
            criterion=[criterion[i]],
            metric=[metric[i]],
            train_loader=[train_loader[i]],
            val_loader=[val_loader[i]],
            n_epochs=n_epochs,
            device=device,
            weights=weights,
            watcher=watcher,
            save_best_path=save_best_path,
            overall_config=overall_config,
            dataloader_mode=dataloader_mode,
            metric_comparator=metric_comparator,
            verbose=verbose,
            skip_train_eval=skip_train_eval,
            best_metric=best_metric,
            prior_scores=prior_scores,
            skip_optimizer_step=skip_optimizer_step,
        )
    return result, best_metric
