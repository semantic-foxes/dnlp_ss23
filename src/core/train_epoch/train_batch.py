import torch
from torch import nn


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

        if task == 'paraphrase_classifier':
            # predictions with embeddings
            return model(task, ids_1, attention_masks_1, ids_2, attention_masks_2, True)
        
        predictions = model(task, ids_1, attention_masks_1, ids_2, attention_masks_2)


    else:
        raise NotImplementedError

    return (predictions, [])


def train_one_batch_multitask(
        model: nn.Module,
        batch: torch.tensor,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        task: str,
        is_optimizer_step: bool = True,
        loss_divisor: int = 1,
        cosine_loss = None
):
    predictions, embeddings = _batch_forward(batch, model, task, device)
    targets = batch['targets'].to(device)
    
    loss = criterion(predictions, targets).sum() / loss_divisor
    if cosine_loss and task == 'paraphrase_classifier':
        print(embeddings)
        print(targets)
        # targets should be -1,1
        loss = loss + cosine_loss(*embeddings, 2*targets-1) / 2
        
    loss.backward()

    if is_optimizer_step:
        optimizer.step()
        optimizer.zero_grad()
