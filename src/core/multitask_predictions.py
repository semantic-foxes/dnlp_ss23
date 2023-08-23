import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn



@torch.no_grad()
def generate_predictions_multitask(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        dataloader_message: str = 'test'
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
    task =  dataloader.dataset.task
    model.eval()
    result = pd.DataFrame(columns=['prediction'])
    progress_bar = tqdm(dataloader,
                        desc=f'Making predictions on {task}')

    

    for i, batch in enumerate(progress_bar):
        if task == 'sentiment':
                ids, attention_masks = batch['token_ids'], batch['attention_masks']

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

                predictions = model(task, ids_1, attention_masks_1, ids_2, attention_masks_2)

        else:
            raise NotImplementedError


        predictions = predictions.cpu().numpy()
        if task == 'sentiment' or task == 'paraphrase_classifier':
            predictions = np.argmax(predictions, axis=1)
        predictions = predictions.flatten()

        new_dataframe = pd.DataFrame(
            {'prediction': predictions}
        )

        result = pd.concat([result, new_dataframe])

    return result
