"""Functions related to the dashboard / task execution."""

from functools import partial
from importlib.resources import files
from typing import Any, Callable

import streamlit as st
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit.delta_generator import DeltaGenerator

import config
from config import task
from diffusion_inverse_comparison.config_models import ModelName, ModelConfig
from diffusion_inverse_comparison.conditioning_methods import (
    ConditioningMethod,
    ConditioningMethod_,
    get_conditioning_method,
)
from diffusion_inverse_comparison.dataset import DatasetType, get_dataloader, get_dataset
from diffusion_inverse_comparison.noise import get_noise
from diffusion_inverse_comparison.operator import Operator, get_operator
from diffusion_inverse_comparison.sampler import Sampler, GaussianDiffusion, create_sampler
from diffusion_inverse_comparison.unet import UNetModel, create_model
from diffusion_inverse_comparison.utils.image import clear_color, mask_generator
from diffusion_inverse_comparison.utils.logger import get_logger

logger = get_logger()


def get_device() -> torch.device:
    """Automatically gets torch device to use (CUDA or CPU), picking CUDA when available.

    Returns:
        torch.device: Chosen torch device.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


@st.cache_resource(hash_funcs={ModelConfig: str})
def load_model(model_config: ModelConfig, device: torch.device) -> UNetModel:
    """Load UNet backwards process model.

    Args:
        model_config (ModelConfig): Config for the model.
        device (torch.device): Device on which to load model

    Returns:
        UNetModel: Loaded model.
    """
    return create_model(model_config).to(device).eval()


def load_sampler(**kwargs: Any) -> GaussianDiffusion:
    """Load diffusion sampler.

    Returns:
        GaussianDiffusion: Loaded sampler model.
    """
    diffusion_conf_path = files(config) / "diffusion.yaml"

    with diffusion_conf_path.open() as f:
        diffusion_config = yaml.load(f, Loader=yaml.FullLoader) | kwargs

    return create_sampler(**diffusion_config)


def load_task_config(task_name: Operator) -> dict[str, Any]:
    """Loads chosen task config.

    Args:
        task_name (Operator): Name of task.

    Returns:
        dict[str, Any]: Task config dictionary.
    """
    task_conf_path = files(task) / f"{task_name}.yaml"

    with task_conf_path.open() as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_sample_fn(
    cond_method: ConditioningMethod_,
    sampler: GaussianDiffusion,
    model: UNetModel,
    mask: torch.Tensor | None = None,
) -> partial:
    """_summary_

    Args:
        cond_method (ConditioningMethod_):
        sampler (Sampler): Diffusion sampler.
        model (UNetModel): Backwards process model.
        mask (torch.Tensor | None, optional): Mask to apply to measurement. Defaults to None.

    Returns:
        partial: Partially applied sampling function.
    """
    measurement_cond_fn = (
        cond_method.conditioning if mask is None else partial(cond_method.conditioning, mask=mask)
    )
    return partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)


def load(
    model_name: ModelName | None,
    task_name: Operator | None,
    sampler_name: Sampler | None,
    conditioning_method_name: ConditioningMethod | None,
    source_name: DatasetType | None,
    timestep_respacing: int,
    uploaded_img_buffer: UploadedFile | None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable]]:
    """Load requisites for dasbhoard.

    Args:
        model_name (ModelName | None): Model name.
        task_name (Operator | None): Task name.
        sampler_name (Sampler | None): Sampler name.
        conditioning_method_name (ConditioningMethod | None): Conditioning method name.
        source_name (DatasetType | None): Source name.
        timestep_respacing (int): Amount of timestep respacing (for DDIM).
        uploaded_img_buffer (UploadedFile | None): Buffer of uploaded image, if source is own.

    Raises:
        ValueError: Task can't be None.
        ValueError: Sampler can't be None.
        ValueError: Conditioning method can't be None.
        ValueError: Source can't be None.

    Returns:
        list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable]]: List of samples with each
            being a tuple comprising the original image (uploaded or from dataset), the noised
            image (with transformations applied), the starting noise from which to begin the
            backwards process, and the sampling function associated with the sample.
    """

    if model_name is None:
        raise ValueError("Bad model name")
    if task_name is None:
        raise ValueError("Bad task")
    if sampler_name is None:
        raise ValueError("Bad sampler")
    if conditioning_method_name is None:
        raise ValueError("Bad conditioning method")
    if source_name is None:
        raise ValueError("Bad source")

    logger.info(f"Using model: {model_name}")
    logger.info(f"Using task: {task_name}")
    logger.info(f"Using sampler: {sampler_name}")
    logger.info(f"Using conditioning method: {conditioning_method_name}")
    logger.info(f"Using source: {source_name}")
    logger.info(f"Using timestep respacing: {timestep_respacing}")

    device = get_device()
    logger.info(f"Using device: {device}")

    model_config = ModelConfig.load(model_name)
    model = load_model(model_config, device)
    logger.info("Loaded model...")

    sampler = load_sampler(sampler=sampler_name, timestep_respacing=timestep_respacing)
    logger.info("Loaded sampler...")

    task_config = load_task_config(task_name)
    cond_config = task_config["conditioning"]
    measure_config = task_config["measurement"]
    operator_config = measure_config["operator"]
    logger.info("Loaded task config...")

    operator = get_operator(device=device, **operator_config)
    logger.info("Loaded operator...")

    noiser = get_noise(**measure_config["noise"])
    logger.info("Loaded noiser...")

    cond_method = get_conditioning_method(
        conditioning_method_name, operator, noiser, **cond_config["params"]
    )
    base_sample_fn = get_sample_fn(cond_method, sampler, model)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    out: list[tuple[Any, ...]] = []
    if source_name == DatasetType.OWN:
        if uploaded_img_buffer is None:
            return out

        img = Image.open(uploaded_img_buffer).convert("RGB")
        ref_img = transform(img).to(device)[None, ...]  # add dummy batch dim

        y_n = noiser(ref_img)

        out_shape = (
            y_n.shape
            if task_name != Operator.SUPER_RESOLUTION
            else measure_config["operator"]["in_shape"]
        )
        x_start = torch.randn(out_shape, device=device, requires_grad=True)

        out.append((ref_img.clone(), y_n.clone(), x_start, base_sample_fn))
    else:
        dataset = get_dataset(name=source_name, transforms=transform)
        loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

        if task_name == Operator.INPAINTING:
            mask_gen = mask_generator(**measure_config["mask_opt"])

        for ref_img in iter(loader):
            ref_img = ref_img.to(device)

            if task_name == Operator.INPAINTING:
                mask = mask_gen(ref_img)[:, 0, ...].unsqueeze(dim=0)
                sample_fn = get_sample_fn(cond_method, sampler, model, mask)
                y = operator.forward(ref_img, mask=mask)
            else:
                sample_fn = base_sample_fn
                y = operator.forward(ref_img)

            y_n = noiser(y)

            out_shape = (
                y_n.shape
                if task_name != Operator.SUPER_RESOLUTION
                else measure_config["operator"]["in_shape"]
            )
            x_start = torch.randn(out_shape, device=device, requires_grad=True)

            out.append((ref_img.clone(), y_n.clone(), x_start, sample_fn))

    logger.info("Finished loading!")
    return out


def run(
    loaded: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable]],
    col_out: DeltaGenerator,
    progress_caption: str,
) -> None:
    """Runs inverse task for each sample of loaded.

    Note:
        Sets and repeatedly updates (during sampling) the output image in the `col_out` streamlit
        column as a side-effect.

    Args:
        loaded (list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable]]): List of samples
            with each being a tuple comprising the original image (uploaded or from dataset), the
            noised image (with transformations applied), the starting noise from which to begin the
            backwards process, and the sampling function associated with the sample.
        col_out (DeltaGenerator): The output streamlit column reference.
        progress_caption (str): Caption for streamlit progress bar.
    """
    logger.info("Running...")
    for _, y_n, x_start, sample_fn in loaded:
        with col_out:
            placeholder = st.empty()
            prog_bar = st.progress(0, text=progress_caption)

            for sample, _, percent_complete in sample_fn(x_start=x_start, measurement=y_n):
                with placeholder.container():
                    st.image(clear_color(sample.clone()), use_column_width=True)
                prog_bar.progress(percent_complete, text=progress_caption)

            prog_bar.empty()
    logger.info("Run finished!")
