import os
from typing import Union

import torch
from torch import nn
from src.utils.logger_config import logger


def generate_device(use_cuda: bool) -> torch.device:
    """
    Provides a devices based on either you want to use `cuda` or not.

    Parameters
    ----------
    use_cuda : bool
        If using a `cuda` device if possible is required.

    Returns
    -------
    device : torch.device
        The available device for further usage.
    """
    if use_cuda:
        if not torch.cuda.is_available():
            message = (
                "WARNING: CUDA is not available while being asked for it. "
                "Falling back to CPU."
            )
            logger.warn(message)
            device = torch.device("cpu")

        else:
            device = torch.device("cuda:0")

    else:
        device = torch.device("cpu")

    return device


def get_device(obj: Union[nn.Module, torch.Tensor]):
    if type(obj) == torch.Tensor:
        device = obj.device
    else:
        try:
            device = next(obj.parameters()).device
        except AttributeError as error:
            logger.error(error)
            raise error

    return device


def save_state(
    model: nn.Module, optimizer: torch.optim.Optimizer, config: dict, filepath: str
):
    """
    Saves the model state, the optimizer state and the config
    at the given path. The path is expected to have the files itself as well.

    Parameters
    ----------
    model : nn.Module
    optimizer : torch.optim.Optimizer
    config : dict
    filepath : str
        Path to the saved file. Should specify the file as well.
    """

    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)

    logger.info(f"Saving the model to {filepath}.")

    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "config": config,
    }

    torch.save(save_info, filepath)
    logger.info(
        f"Successfully saved the model, optimizer " f"and config to {filepath}."
    )
