import torch
from torch import nn

def train_one_batch_multitask(
        model: nn.Module,
        batch: torch.tensor,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        task: str
):
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