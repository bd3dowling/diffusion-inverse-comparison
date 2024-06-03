"""Pydantic models for configs."""

from importlib.resources import files
from importlib.abc import Traversable
from typing import Any

import yaml
from pydantic import BaseModel, PositiveInt
from strenum import StrEnum

import config
import config.model
import data.model_checkpoint
import data.samples
from diffusion_inverse_comparison.conditioning_methods import ConditioningMethodName
from diffusion_inverse_comparison.operator import Operator
from diffusion_inverse_comparison.sampler import Sampler
from diffusion_inverse_comparison.dataset import DatasetType


class ModelName(StrEnum):
    FFHQ = "ffhq"
    IMAGENET = "imagenet"


class DashboardConfig(BaseModel, protected_namespaces=()):
    dashboard_title: str
    task_caption: str
    task_help: str
    sampler_caption: str
    sampler_help: str
    conditioning_method_caption: str
    conditioning_method_help: str
    model_caption: str
    model_help: str
    source_caption: str
    source_help: str
    ts_respacing_caption: str
    ts_respacing_help: str
    run_label: str
    run_help: str
    stop_label: str
    stop_help: str
    img_upload_prompt: str
    ref_col_header: str
    ref_col_help: str
    input_col_header: str
    input_col_help: str
    output_col_header: str
    output_col_help: str
    progress_caption: str
    ts_respacing_vals: list[PositiveInt]
    task_label_map: dict[Operator, str]
    sampler_label_map: dict[Sampler, str]
    conditioning_method_label_map: dict[ConditioningMethodName, str]
    source_label_map: dict[DatasetType, str]
    model_label_map: dict[ModelName, str]

    @classmethod
    def load(cls) -> "DashboardConfig":
        dashboard_conf_path = files(config) / "dashboard.yaml"

        with dashboard_conf_path.open() as f:
            dashboard_conf_raw: dict[str, Any] = yaml.safe_load(f)

        return cls(**dashboard_conf_raw)


class ModelConfig(BaseModel, frozen=True):
    image_size: int
    num_channels: int
    num_res_blocks: int
    channel_mult: str
    learn_sigma: bool
    class_cond: bool
    use_checkpoint: bool
    attention_resolutions: str | int
    num_heads: int
    num_head_channels: int
    num_heads_upsample: int
    use_scale_shift_norm: bool
    dropout: int
    resblock_updown: bool
    use_new_attention_order: bool
    _model_name: ModelName  # Not validated

    @classmethod
    def load(cls, model_name: ModelName) -> "ModelConfig":
        model_conf_path = files(config.model) / f"{model_name}.yaml"

        with model_conf_path.open() as f:
            model_conf_raw: dict[str, Any] = yaml.safe_load(f)

        instance = cls(**model_conf_raw)
        instance._model_name = model_name

        return instance

    @property
    def model_checkpoint_path(self) -> Traversable:
        return files(data.model_checkpoint) / f"{self._model_name}.pt"
