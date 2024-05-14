"""Noise type definitions and registry."""

from abc import ABC, abstractmethod
from torchvision import torch
from enum import StrEnum
import numpy as np


__NOISE__ = {}


class Noise(StrEnum):
    CLEAN = "clean"
    GAUSSIAN = "gaussian"
    POISSON = "poisson"


def register_noise(name: Noise):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls

    return wrapper


def get_noise(name: Noise, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser


class _Noise(ABC):
    def __call__(self, data):
        return self.forward(data)

    @abstractmethod
    def forward(self, data):
        raise NotImplementedError


@register_noise(name=Noise.CLEAN)
class Clean(_Noise):
    def forward(self, data):
        return data


@register_noise(name=Noise.GAUSSIAN)
class GaussianNoise(_Noise):
    def __init__(self, sigma):
        self.sigma = sigma

    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name=Noise.POISSON)
class PoissonNoise(_Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        """
        Follow skimage.util.random_noise.
        """
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)
