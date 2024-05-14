# diffusion_inverse_comparison

This is where the core application code lives. The following roughly summarises each
module/package's contents:

- [`app_functions.py`](./app_functions.py):
  Functions related to the dashboard / task execution.
- [`conditioning_methods.py`](./conditioning_methods.py):
  Conditioning method definitions and registry.
- [`config_models.py`](./config_models.py):
  Pydantic models for configs.
- [`dataset.py`](./dataset.py):
  Dataset definitions and registry.
- [`noise.py`](./noise.py):
  Noise type definitions and registry.
- [`operator.py`](./operator.py):
  Operator/task definitions and registry.
- [`posterior_mean_variance.py`](./posterior_mean_variance.py):
  Posterior mean and variance processor definitions and registries.
- [`sampler.py`](./sampler.py):
  Sampler definitions and registry.
- [`unet.py`](./unet.py):
  PyTorch model code.
- [`utils.py`](./utils/):
  Various utility function modules.
