# models

This is where PyTorch model checkpoints should be downloaded to. Currently, the application is
setup to only support the pre-trained FFHQ_10M and ImageNet256 model described in the
[original repo](https://github.com/DPS2022/diffusion-posterior-sampling/tree/main).
The directory setup as a python package so that the checkpoint files can be loaded with
`importlib.resources.files`.
