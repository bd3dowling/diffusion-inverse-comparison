"""Utility functions for working with numpy and torch arrays/tensors."""

import numpy as np
import torch


def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)


def expand_as(array, target):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array)
    elif isinstance(array, np.floating):
        array = torch.tensor([array])

    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)

    return array.expand_as(target).to(target.device)
