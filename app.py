"""Main dashboard code."""

from functools import partial
from importlib.resources import files
from typing import Any

import streamlit as st
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile

import configs
from configs import task_configs
from data import samples
from src.config_models import DashboardConfig, SourceOption
from src.conditioning_methods import ConditioningMethod, get_conditioning_method
from src.dataset import Dataset, get_dataloader, get_dataset
from src.noise import get_noise
from src.operator import Operator, get_operator
from src.sampler import Sampler, create_sampler
from src.unet import create_model
from src.utils.image import clear_color, mask_generator
from src.utils.logger import get_logger

logger = get_logger()
dashboard_config = DashboardConfig.load()

if "running" not in st.session_state:
    st.session_state["running"] = False


def run_callback():
    st.session_state["running"] = True


def stop_callback():
    st.session_state["running"] = False


def get_device():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


@st.cache_resource
def load_model(device):
    model_conf_path = files(configs) / "model_config.yaml"

    with model_conf_path.open() as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    return create_model(**model_config).to(device).eval()


def load_sampler(**kwargs):
    diffusion_conf_path = files(configs) / "diffusion_config.yaml"

    with diffusion_conf_path.open() as f:
        diffusion_config = yaml.load(f, Loader=yaml.FullLoader) | kwargs

    return create_sampler(**diffusion_config)


def load_task_config(task_name: Operator):
    task_conf_path = files(task_configs) / f"{task_name}_config.yaml"

    with task_conf_path.open() as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_sample_fn(cond_method, sampler, model, mask=None):
    measurement_cond_fn = (
        cond_method.conditioning if mask is None else partial(cond_method.conditioning, mask=mask)
    )
    return partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)


def load(
    task_name: Operator | None,
    sampler_name: Sampler | None,
    conditioning_method_name: ConditioningMethod | None,
    source: SourceOption | None,
    timestep_respacing: int,
    uploaded_img_buffer: UploadedFile | None,
):
    if task_name is None:
        raise ValueError("Bad task")
    if sampler_name is None:
        raise ValueError("Bad sampler")
    if conditioning_method_name is None:
        raise ValueError("Bad conditioning method")
    if source is None:
        raise ValueError("Bad source")

    logger.info(f"Loading task: {task_name}")
    logger.info(f"Using sampler: {sampler_name}")
    logger.info(f"Using conditioning method: {conditioning_method_name}")
    logger.info(f"Using source: {source}")
    logger.info(f"Using timestep respacing: {timestep_respacing}")

    device = get_device()
    logger.info(f"Using device: {device}")

    model = load_model(device)
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
    if source == SourceOption.OWN:
        if uploaded_img_buffer is None:
            return out

        img = Image.open(uploaded_img_buffer).convert("RGB")
        ref_img = transform(img).to(device)[None, ...]

        y_n = noiser(ref_img)

        out_shape = (
            y_n.shape
            if task_name != Operator.SUPER_RESOLUTION
            else measure_config["operator"]["in_shape"]
        )
        x_start = torch.randn(out_shape, device=device).requires_grad_()

        out.append((ref_img.clone(), y_n.clone(), x_start, base_sample_fn))
    else:
        dataset = get_dataset(name=Dataset.FFHQ, root=str(files(samples)), transforms=transform)
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
            x_start = torch.randn(out_shape, device=device).requires_grad_()

            out.append((ref_img.clone(), y_n.clone(), x_start, sample_fn))

    logger.info("Finished loading!")
    return out


def run(loaded, col_out):
    logger.info("Running...")
    for _, y_n, x_start, sample_fn in loaded:
        with col_out:
            placeholder = st.empty()
            prog_bar = st.progress(0, text=dashboard_config.progress_caption)

            for sample, _, percent_complete in sample_fn(x_start=x_start, measurement=y_n):
                with placeholder.container():
                    st.image(clear_color(sample.clone()), use_column_width=True)
                prog_bar.progress(percent_complete, text=dashboard_config.progress_caption)

            prog_bar.empty()
    logger.info("Run finished!")
    st.session_state["running"] = False


#############
# DASHBOARD #
#############

st.title(dashboard_config.dashboard_title)

# - Controls
col_task, col_sampler, col_cond, col_src = st.columns(4)

with col_task:
    task_name = st.radio(
        label=dashboard_config.task_caption,
        options=dashboard_config.task_label_map.keys(),
        format_func=lambda key: dashboard_config.task_label_map[key],
        on_change=stop_callback,
    )

with col_sampler:
    sampler_name = st.radio(
        label=dashboard_config.sampler_caption,
        options=dashboard_config.sampler_label_map.keys(),
        format_func=lambda key: dashboard_config.sampler_label_map[key],
        on_change=stop_callback,
    )

with col_cond:
    conditioning_method_name = st.radio(
        label=dashboard_config.conditioning_method_caption,
        options=dashboard_config.conditioning_method_label_map.keys(),
        format_func=lambda key: dashboard_config.conditioning_method_label_map[key],
        on_change=stop_callback,
    )

with col_src:
    source = st.radio(
        label=dashboard_config.source_caption,
        options=dashboard_config.source_label_map.keys(),
        format_func=lambda key: dashboard_config.source_label_map[key],
        on_change=stop_callback,
    )

if sampler_name == Sampler.DDIM:
    timestep_respacing = st.select_slider(
        label=dashboard_config.ts_respacing_caption,
        options=dashboard_config.ts_respacing_vals,
        value=dashboard_config.ts_respacing_vals[-1],
        on_change=stop_callback,
    )
else:
    timestep_respacing = dashboard_config.ts_respacing_vals[-1]

if not st.session_state["running"]:
    st.button(
        dashboard_config.run_label, type="primary", use_container_width=True, on_click=run_callback
    )
else:
    st.button(
        dashboard_config.stop_label,
        type="secondary",
        use_container_width=True,
        on_click=stop_callback,
    )

if source == SourceOption.OWN:
    col_in, col_out = st.columns(2)
    with col_in:
        st.header(dashboard_config.input_col_header)
        img_placeholder = st.empty()
        uploaded_img_buffer = st.file_uploader(
            dashboard_config.img_upload_prompt, type=["png", "jpg"]
        )
    with col_out:
        st.header(dashboard_config.output_col_header)
else:
    col_ref, col_in, col_out = st.columns(3)
    with col_ref:
        st.header(dashboard_config.ref_col_header)
    with col_in:
        st.header(dashboard_config.input_col_header)
    with col_out:
        st.header(dashboard_config.output_col_header)
    uploaded_img_buffer = None

# - Data grid
loaded = load(
    task_name,
    sampler_name,
    conditioning_method_name,
    source,
    timestep_respacing,
    uploaded_img_buffer,
)

if source == SourceOption.OWN:
    if uploaded_img_buffer:
        ref_img = loaded[0][0]
        with img_placeholder.container():
            st.image(clear_color(ref_img), use_column_width=True)
else:
    for ref_img, y_n, _, _ in loaded:
        with col_ref:
            st.image(clear_color(ref_img), use_column_width=True)
        with col_in:
            st.image(clear_color(y_n), use_column_width=True)

if st.session_state["running"]:
    run(loaded, col_out)
