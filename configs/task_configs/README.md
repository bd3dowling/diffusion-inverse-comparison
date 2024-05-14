# task_configs

This is where all task configuration YAMLs live. For now, these are the only way of editing the task
definitions. In future, these should be treated as base configurations which can be overriden by
controls on the dashboard. The directory is setup as a python package (module)
so that the YAMLs can be loaded with `importlib.resources.files`. The files are as follows:

- [`gaussian_blur_config.yaml`](./gaussian_blur_config.yaml):
  Config for [super resolution](https://en.wikipedia.org/wiki/Super-resolution_imaging) task.
- [`inpainting_config.yaml`](./inpainting_config.yaml):
  Config for [motion deblur](https://en.wikipedia.org/wiki/Motion_blur) task.
- [`motion_blur_config.yaml`](./motion_blur_config.yaml):
  Config for [gaussian deblur](https://en.wikipedia.org/wiki/Gaussian_blur) task.
- [`super_resolution_config.yaml`](./super_resolution_config.yaml):
  Config for [inpainting](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint) task.
