from enum import StrEnum

import yaml
import configs
from typing import Any
from importlib.resources import files
from pydantic import BaseModel, PositiveInt
from diffusion_inverse_comparison.operator import Operator
from diffusion_inverse_comparison.sampler import Sampler
from diffusion_inverse_comparison.conditioning_methods import ConditioningMethod


class SourceOption(StrEnum):
    FFHQ = "ffhq"
    OWN = "own"


class DashboardConfig(BaseModel):
    dashboard_title: str
    task_caption: str
    sampler_caption: str
    conditioning_method_caption: str
    source_caption: str
    ts_respacing_caption: str
    run_label: str
    stop_label: str
    img_upload_prompt: str
    ref_col_header: str
    input_col_header: str
    output_col_header: str
    progress_caption: str
    ts_respacing_vals: list[PositiveInt]
    task_label_map: dict[Operator, str]
    sampler_label_map: dict[Sampler, str]
    conditioning_method_label_map: dict[ConditioningMethod, str]
    source_label_map: dict[SourceOption, str]

    @classmethod
    def load(cls) -> "DashboardConfig":
        dashboard_conf_path = files(configs) / "dashboard_config.yaml"

        with dashboard_conf_path.open() as f:
            dashboard_conf_raw: dict[str, Any] = yaml.safe_load(f)

        return cls(**dashboard_conf_raw)
