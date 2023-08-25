from typing import Callable, List
from tqdm import tqdm

import torch
from torch import nn



@torch.no_grad()
def evaluate_model_multitask(
        model: nn.Module,
        eval_dataloaders: List[torch.utils.data.DataLoader],
        device: torch.device,
        metrics: List[Callable],
        criterions: List[torch.nn.Module] = None,
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

    loss = {}
    metric = {}

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
            loss[task] = running_loss / len(dataloader.dataset)

        metric[task] = running_metric / len(dataloader.dataset)

    return {'metric': metric, 'loss': loss}


def evaluation_message(result: dict)->str:
    metric = result['metric']
    loss = result['loss']

    message = 'metric: '
    metric_strings = [f'{key}: {value:.3f}' for key, value in metric.items()]
    message += ', '.join(metric_strings)
    message += ' loss: '
    loss_strings = [f'{key}: {value:.3f}' for key, value in loss.items()]
    message += ', '.join(loss_strings)

    return message

def sum_comparator(current:dict, best:dict)->bool:
    return sum(current.values()) > sum(best.values())
