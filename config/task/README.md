# config.task

This is where all task configuration YAMLs live. For now, these are the only way of editing the task
definitions. In future, these should be treated as base configurations which can be overriden by
controls on the dashboard. The directory is setup as a python package (module)
so that the YAMLs can be loaded with `importlib.resources.files`. The files are as follows:

- [`gaussian_blur.yaml`](./gaussian_blur.yaml):
  Config for [super resolution](https://en.wikipedia.org/wiki/Super-resolution_imaging) task.
- [`inpainting.yaml`](./inpainting.yaml):
  Config for [motion deblur](https://en.wikipedia.org/wiki/Motion_blur) task.
- [`motion_blur.yaml`](./motion_blur.yaml):
  Config for [gaussian deblur](https://en.wikipedia.org/wiki/Gaussian_blur) task.
- [`super_resolution.yaml`](./super_resolution.yaml):
  Config for [inpainting](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint) task.
