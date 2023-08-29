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
        watcher: Union[str, None] = None,
        verbose: bool = True,
        save_best_path: str = None,
        overall_config: dict = None,
        metric_comparator: Callable[[dict, dict], bool] = sum_comparator,
        data_combine: str = 'sequential',
        skip_train_eval: int = 1,
        best_metric: dict = {},
        result: List = []
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
            watcher=watcher,
            save_best_path=save_best_path,
            overall_config=overall_config,
            data_combine=data_combine,
            metric_comparator=metric_comparator,
            verbose=verbose,
            skip_train_eval=skip_train_eval,
            best_metric=best_metric,
            result=result
        )
    return result, best_metric
