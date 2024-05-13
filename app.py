from enum import StrEnum
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
from src.conditioning_methods import ConditioningMethod, get_conditioning_method
from src.dataloader import get_dataloader, get_dataset
from src.noise import get_noise
from src.operator import Operator, get_operator
from src.sampler import Sampler, create_sampler
from src.unet import create_model
from src.utils.image import clear_color, mask_generator
from src.utils.logger import get_logger

logger = get_logger()


class SourceOption(StrEnum):
    FFHQ = "ffhq"
    OWN = "own"


DASHBOARD_TITLE = "Diffusion Models for Inverse Problem Solving"

TASK_CAPTION = "Task:"
SAMPLER_CAPTION = "Sampler:"
CONDITIONING_METHOD_CAPTION = "Condition method:"
SOURCE_CAPTION = "Test image(s) source:"

TS_RESPACING_CAPTION = "Timespace respacing:"

RUN_LABEL = "Run"
STOP_LABEL = "Stop"
IMG_UPLOAD_PROMPT = "Upload an image:"
REF_COL_HEADER = "Reference"
INPUT_COL_HEADER = "Input"
OUTPUT_COL_HEADER = "Output"
PROGRESS_CAPTION = "Denoising..."

TS_RESPACING_VALS = (100, 250, 500, 1000)
TASK_LABEL_MAP = {
    Operator.SUPER_RESOLUTION: "Super resolution",
    Operator.MOTION_BLUR: "Motion deblur",
    Operator.INPAINTING: "Inpainting",
    Operator.GAUSSIAN_BLUR: "Gaussian deblur",
}
SAMPLER_LABEL_MAP = {Sampler.DDPM: "DDPM", Sampler.DDIM: "DDIM"}
CONDITIONING_METHOD_LABEL_MAP = {
    ConditioningMethod.POSTERIOR_SAMPLING: "DPS",
    ConditioningMethod.PROJECTION: "Projection",
    ConditioningMethod.VANILLA: "No conditioning",
}
SOURCE_LABEL_MAP = {SourceOption.FFHQ: "FFHQ samples", SourceOption.OWN: "Own"}


# State initialization
if "running" not in st.session_state:
    st.session_state["running"] = False


def run_callback():
    st.session_state["running"] = True


def stop_callback():
    st.session_state["running"] = False


@st.cache_resource
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

    device = get_device()
    model = load_model(device)
    sampler = load_sampler(sampler=sampler_name, timestep_respacing=timestep_respacing)
    task_config = load_task_config(task_name)
    cond_config = task_config["conditioning"]
    measure_config = task_config["measurement"]
    operator_config = measure_config["operator"]

    operator = get_operator(device=device, **operator_config)
    noiser = get_noise(**measure_config["noise"])
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
        dataset = get_dataset(name="ffhq", root=str(files(samples)), transforms=transform)
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

    return out


def run(loaded, col_out):
    for _, y_n, x_start, sample_fn in loaded:
        with col_out:
            placeholder = st.empty()
            prog_bar = st.progress(0, text=PROGRESS_CAPTION)

            for sample, _, percent_complete in sample_fn(x_start=x_start, measurement=y_n):
                with placeholder.container():
                    st.image(clear_color(sample.clone()), use_column_width=True)
                prog_bar.progress(percent_complete, text=PROGRESS_CAPTION)

            prog_bar.empty()

    st.session_state["running"] = False


# Dashboard

st.title(DASHBOARD_TITLE)

# - Controls
col_task, col_sampler, col_cond, col_src = st.columns(4)

with col_task:
    task_name = st.radio(
        label=TASK_CAPTION,
        options=TASK_LABEL_MAP.keys(),
        format_func=lambda key: TASK_LABEL_MAP[key],
        on_change=stop_callback,
    )

with col_sampler:
    sampler_name = st.radio(
        label=SAMPLER_CAPTION,
        options=SAMPLER_LABEL_MAP.keys(),
        format_func=lambda key: SAMPLER_LABEL_MAP[key],
        on_change=stop_callback,
    )

with col_cond:
    conditioning_method_name = st.radio(
        label=CONDITIONING_METHOD_CAPTION,
        options=CONDITIONING_METHOD_LABEL_MAP.keys(),
        format_func=lambda key: CONDITIONING_METHOD_LABEL_MAP[key],
        on_change=stop_callback,
    )

with col_src:
    source = st.radio(
        label=SOURCE_CAPTION,
        options=SOURCE_LABEL_MAP.keys(),
        format_func=lambda key: SOURCE_LABEL_MAP[key],
        on_change=stop_callback,
    )

if sampler_name == Sampler.DDIM:
    timestep_respacing = st.select_slider(
        label=TS_RESPACING_CAPTION,
        options=TS_RESPACING_VALS,
        value=TS_RESPACING_VALS[-1],
        on_change=stop_callback,
    )
else:
    timestep_respacing = TS_RESPACING_VALS[-1]

if not st.session_state["running"]:
    st.button(RUN_LABEL, type="primary", use_container_width=True, on_click=run_callback)
else:
    st.button(STOP_LABEL, type="secondary", use_container_width=True, on_click=stop_callback)

if source == SourceOption.OWN:
    col_in, col_out = st.columns(2)
    with col_in:
        st.header(INPUT_COL_HEADER)
        img_placeholder = st.empty()
        uploaded_img_buffer = st.file_uploader(IMG_UPLOAD_PROMPT, type=["png", "jpg"])
    with col_out:
        st.header(OUTPUT_COL_HEADER)
else:
    col_ref, col_in, col_out = st.columns(3)
    with col_ref:
        st.header(REF_COL_HEADER)
    with col_in:
        st.header(INPUT_COL_HEADER)
    with col_out:
        st.header(OUTPUT_COL_HEADER)
    uploaded_img_buffer = None

# - Data grid
loaded = load(task_name, sampler_name, conditioning_method_name, source, timestep_respacing, uploaded_img_buffer)

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
