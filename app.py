"""Main dashboard code."""

import streamlit as st

from diffusion_inverse_comparison.config_models import DashboardConfig, SourceOption
from diffusion_inverse_comparison.sampler import Sampler
from diffusion_inverse_comparison.utils.image import clear_color
from diffusion_inverse_comparison.app_functions import load, run


# Initialize state variable for tracking run
# NOTE: Use states rather than direct callback for running since makes stopping easier in streamlit
if "running" not in st.session_state:
    st.session_state["running"] = False

# Callbacks


def run_callback() -> None:
    """Callback function for run button.

    Note:
        Sets `running` state variable to `True` as side-effect.
    """
    st.session_state["running"] = True


def stop_callback() -> None:
    """Callback function for stop button or control changes.

    Note:
        Sets `running` state variable to `False` as side-effect.
    """
    st.session_state["running"] = False


# Dashboard

dashboard_config = DashboardConfig.load()

st.title(dashboard_config.dashboard_title)

# - Controls
col_task, col_sampler, col_cond = st.columns(3)

# Task selector
with col_task:
    task_name = st.radio(
        label=dashboard_config.task_caption,
        help=dashboard_config.task_help,
        options=dashboard_config.task_label_map.keys(),
        format_func=lambda key: dashboard_config.task_label_map[key],
        on_change=stop_callback,
    )

# Sampler selector
with col_sampler:
    sampler_name = st.radio(
        label=dashboard_config.sampler_caption,
        help=dashboard_config.sampler_help,
        options=dashboard_config.sampler_label_map.keys(),
        format_func=lambda key: dashboard_config.sampler_label_map[key],
        on_change=stop_callback,
    )

# Conditioning method selector
with col_cond:
    conditioning_method_name = st.radio(
        label=dashboard_config.conditioning_method_caption,
        help=dashboard_config.conditioning_method_help,
        options=dashboard_config.conditioning_method_label_map.keys(),
        format_func=lambda key: dashboard_config.conditioning_method_label_map[key],
        on_change=stop_callback,
    )

# Show timestep respacing slider if using DDIM sampler
if sampler_name == Sampler.DDIM:
    timestep_respacing = st.select_slider(
        label=dashboard_config.ts_respacing_caption,
        help=dashboard_config.ts_respacing_help,
        options=dashboard_config.ts_respacing_vals,
        value=dashboard_config.ts_respacing_vals[-1],
        on_change=stop_callback,
    )
else:
    timestep_respacing = dashboard_config.ts_respacing_vals[-1]

# Run and stop buttons
if not st.session_state["running"]:
    st.button(
        label=dashboard_config.run_label,
        help=dashboard_config.run_help,
        type="primary",
        use_container_width=True,
        on_click=run_callback,
    )
else:
    st.button(
        label=dashboard_config.stop_label,
        help=dashboard_config.stop_help,
        type="secondary",
        use_container_width=True,
        on_click=stop_callback,
    )

# - Data grid

# NOTE: Placeholder until adequately fix/test custom uploads...
source = SourceOption.FFHQ

# Set data grid columns and their headers.
if source == SourceOption.OWN:
    # If own source, use two columns for input and output.
    col_in, col_out = st.columns(2)
    with col_in:
        st.header(dashboard_config.input_col_header)

        # NOTE: Use empty placeholder here so that image selector always below existing (if present)
        img_placeholder = st.empty()
        uploaded_img_buffer = st.file_uploader(
            dashboard_config.img_upload_prompt, type=["png", "jpg"]
        )
    with col_out:
        st.header(dashboard_config.output_col_header)
else:
    # If ffhq source, use three columns for reference, input (transformed) and ouput.
    col_ref, col_in, col_out = st.columns(3)
    with col_ref:
        st.header(
            body=dashboard_config.ref_col_header,
            help=dashboard_config.ref_col_help,
        )
    with col_in:
        st.header(
            body=dashboard_config.input_col_header,
            help=dashboard_config.input_col_help,
        )
    with col_out:
        st.header(
            body=dashboard_config.output_col_header,
            help=dashboard_config.output_col_help,
        )
    uploaded_img_buffer = None

# Load image(s) either from upload or dataset samples (and transform in such case).
loaded = load(
    task_name,
    sampler_name,
    conditioning_method_name,
    source,
    timestep_respacing,
    uploaded_img_buffer,
)

# Render loaded image(s) into the data grid.
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

# If `running`, use the loaded sample(s) and their sampling functions to start inverse sampling.
# See NOTE at top for why use states and not direct callbacks.
if st.session_state["running"]:
    run(loaded, col_out, dashboard_config.progress_caption)
    st.session_state["running"] = False
