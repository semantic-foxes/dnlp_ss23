import pandas as pd
from tqdm import tqdm

from typing import List

import torch
from torch import nn

@torch.no_grad()
def generate_predictions(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        dataloader_message: str = 'test',
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

@torch.no_grad()
def generate_predictions_multitask(
        model: nn.Module,
        device: torch.device,
        dataloaders: torch.utils.data.DataLoader,
        filepaths: List[str],
):

    for test_loader, save_path in zip(dataloaders, filepaths):
        predictions = generate_predictions(
            model=model,
            dataloader=test_loader,
            device=device,
            dataloader_message=test_loader.dataset.task,
        )

        predictions.to_csv(save_path)

        
