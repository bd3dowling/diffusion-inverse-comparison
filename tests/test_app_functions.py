import torch

from diffusion_inverse_comparison.app_functions import (
    load,
    load_model,
    load_sampler,
    load_task_config,
)
from diffusion_inverse_comparison.operator import Operator
from diffusion_inverse_comparison.sampler import Sampler
from diffusion_inverse_comparison.conditioning_methods import ConditioningMethod
from diffusion_inverse_comparison.config_models import SourceOption


def test_load_model():
    assert load_model(torch.device("cpu")) is not None


def test_load_sampler():
    assert load_sampler() is not None


def test_load_task_config():
    actual = load_task_config(Operator.SUPER_RESOLUTION)
    expected = {
        "conditioning": {"method": "ps", "params": {"scale": 0.3}},
        "measurement": {
            "operator": {
                "name": "super_resolution",
                "in_shape": (1, 3, 256, 256),
                "scale_factor": 4,
            },
            "noise": {"name": "gaussian", "sigma": 0.05},
        },
    }

    assert actual == expected


def test_load():
    assert load(
        Operator.SUPER_RESOLUTION,
        Sampler.DDPM,
        ConditioningMethod.POSTERIOR_SAMPLING,
        SourceOption.FFHQ,
        1000,
        None,
    ) is not None
