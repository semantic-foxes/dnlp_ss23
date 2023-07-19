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
            message = 'WARNING: CUDA is not available while being asked for it. ' \
                      'Falling back to CPU.'
            logger.warn(message)
            device = torch.device('cpu')

        else:
            device = torch.device('cuda:0')

    else:
        device = torch.device('cpu')

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
