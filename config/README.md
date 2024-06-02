# configs

This is where all base configuration YAMLs live. The directory is setup as a python package (module)
so that the YAMLs can be loaded with `importlib.resources.files`. The files are as follows:

- [`dashboard.yaml`](./dashboard.yaml): Config for dashboard such as titles, labels,
  and option values.
- [`diffusion.yaml`](./diffusion.yaml): Config for the diffusion sampler. At present,
  these should be left untouched; the `timestep_respacing` option is overwritten via a slider on the
  dashboard.
- [`model`](./model/): Configs for the (score) models. Seet its [README](./model/README.md).
- [`task`](./task/): Configs for the tasks. See its [README](./task/README.md).