# configs

This is where all base configuration YAMLs live. The directory is setup as a python package (module)
so that the YAMLs can be loaded with `importlib.resources.files`. The files are as follows:

- [`dashboard_config.yaml`](./dashboard_config.yaml): Config for dashboard such as titles, labels,
  and option values.
- [`diffusion_config.yaml`](./diffusion_config.yaml): Config for the diffusion sampler. At present,
  these should be left untouched; the `timestep_respacing` option is overwritten via a slider on the
  dashboard.
- [`model_config.yaml`](./model_config.yaml): Config for the FFHQ pre-trained model. At present,
  values in here cannot be changed (requires ability to retrain model).
- [`task_configs`](./task_configs/): Configs for the tasks. See its
  [README](./task_configs/README.md).
