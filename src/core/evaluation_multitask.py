from typing import Callable, List
from tqdm import tqdm

import torch
from torch import nn

from src.utils import logger


@torch.no_grad()
def evaluate_model_multitask(
        model: nn.Module,
        eval_dataloaders: List[torch.utils.data.DataLoader],
        device: torch.device,
        metrics: List[Callable],
        criterions: List[torch.nn.Module] = None,
        cosine_loss = None,
        overall_config: dict = {},
        verbose: bool = True,
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

    losses = {}
    metric = {}
    if cosine_loss:
        losses['cosine_similarity'] = 0

    for i, dataloader in enumerate(eval_dataloaders):
        task = dataloader.dataset.task
        preds_all = []
        targets_all = []

        if verbose:
            pbar = tqdm(dataloader, leave=False, desc=f'Evaluating on {task}')
        else:
            pbar = dataloader

        for batch in pbar:
            if task == 'sentiment':
                ids, attention_masks, targets = \
                    batch['token_ids'], batch['attention_masks'], batch['targets']

                ids = ids.to(device)
                attention_masks = attention_masks.to(device)
                targets = targets.to(device)

                predictions = model(task, ids, attention_masks)

            elif task == 'paraphrase_classifier':
                ids_1, attention_masks_1, ids_2, attention_masks_2, targets = \
                    (batch['token_ids_1'], batch['attention_masks_1'],
                     batch['token_ids_2'], batch['attention_masks_2'],
                     batch['targets'])

                ids_1 = ids_1.to(device)
                ids_2 = ids_2.to(device)
                attention_masks_1 = attention_masks_1.to(device)
                attention_masks_2 = attention_masks_2.to(device)
                targets = targets.to(device)

                if cosine_loss:
                    predictions, embeddings = model(task, ids_1, attention_masks_1, ids_2, attention_masks_2, True)
                    losses['cosine_similarity'] += cosine_loss(*embeddings, 2*targets-1) / len(targets)
                else:
                    predictions = model(task, ids_1, attention_masks_1, ids_2, attention_masks_2)
                

            elif task == 'paraphrase_regressor':
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
            preds_all.append(predictions) 
            targets_all.append(targets)
        preds_all = torch.cat(preds_all)
        targets_all = torch.cat(targets_all)

        if criterions:
            losses[task] = criterions[i](preds_all, targets_all).item()

        metric[task] = metrics[i](preds_all, targets_all).item()
    
    results = {'metric': metric, 'loss': losses}
    logger.info(evaluation_message(results))

    return results


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
