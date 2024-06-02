# diffusion-inverse-comparison

A [Streamlit](https://streamlit.io/) app for comparing different inverse problem solving methods
with diffusion models. This was originally made for an assignment for the Data Science module as a
part of my MSc in Statistics. The code was bootstrapped from the
[diffusion-posterior-sampling](https://github.com/DPS2022/diffusion-posterior-sampling) repository,
with much of it either heavily inspired by it or directly lifted from it. The principle purpose of
this app is to provide an interactive environment for experimenting with different diffusion
configurations, images, conditioning methods, etc (i.e. as opposed to the scipt and manual YAML
config paradigm of the original repo).

## Features

The ultimate goal of the project is to support full customisation of experiment parameters. At
present, the following capabilities are supported by the dashboard:

- Inverse task selection; following tasks supported:
    - [Super resolution](https://en.wikipedia.org/wiki/Super-resolution_imaging)
    - [Motion deblur](https://en.wikipedia.org/wiki/Motion_blur)
    - [Gaussian deblur](https://en.wikipedia.org/wiki/Gaussian_blur)
    - [Inpainting](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint)
- Sampler selection; following samplers supported:
    - [DDPM](https://arxiv.org/abs/2006.11239)
    - [DDIM](https://arxiv.org/abs/2010.02502)
- Conditioning method selection; currently only supports:
    - Unconditional sampling
    - [Diffusion Posterior Sampling](https://arxiv.org/abs/2209.14687)
- Timestep respacing (when DDIM selected).

Originally, custom uploaded images were supported but this feature was removed since for it to be
sensible several other customisation options need to be added first. As such, three sample images
from the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) are provided.

The codebase is setup to support different models, but currently only the FFHQ-trained UNet model
provided in the [original repo](https://github.com/DPS2022/diffusion-posterior-sampling) has been
tested (and for which samples are provided). See [Future work](#future-work).

## Project Structure

The structure and naming of top level modules follows the suggestion of
[poetry](https://python-poetry.org/docs/basic-usage/) (which we use for dependency management).
Each of the below directories has their own `README`, but roughly their contents are as follows:

- [`config`](/config/): Where all configuration YAMLs live.
- [`data`](/data/): Where all resources, such as model checkpoints and image samples, live.
- [`diffusion_inverse_comparison`](/diffusion_inverse_comparison): Where main application code lives.
- [`external`](/external/): Where external, non-packaged dependencies live.
- [`tests`](/tests/): Where code tests live.
- [`.github`](/.github/): Where configuration/workflow YAMLs live for [Github Actions](https://docs.github.com/en/actions).

The following top-level files serve the following purposes:

- [`app.py`](/app.py): Principle entrypoint and dashboard design code for the Streamlit app.
- [`colab_tunnel.ipynb`](/colab_tunnel.ipynb): Jupyter notebook to be opened in
  [Google Colab](https://colab.research.google.com/) for running app. See [Deployment](#google-colab-recommended)

## Style

To _enforce_ code-style and standards, we use:

- [YAMLlint](https://github.com/adrienverge/yamllint)
- [ruff](https://docs.astral.sh/ruff/)

These are enforced with a [Github Actions](https://docs.github.com/en/actions) (see
[lint.yml](.github/workflows/lint.yml)).

Newly added and refactored code should use the
[Google docstring style](https://google.github.io/styleguide/pyguide.html). As more code and more
code is refactored as this project progresses, additional `pylint` rules will be added to the
[`pyproject.toml`](/pyproject.toml) for `ruff` to check and enforce.

## Deployment

### Google Colab (Recommended)

Given the scale of the models and the computational requirements, the app (realistically) needs to
be run on a machine with a powerful GPU. The provided [Jupyter notebook](/colab_tunnel.ipynb) has
been setup so that running it will install the project into the Colab environment, download the
pre-trained model checkpoint file, and install [localtunnel](https://theboroer.github.io/localtunnel-www/) allowing the app/dashboard to be deployed and accessible over the internet while
using Colab resources. After opening the notebook in Colab, simply:

- Choose a T4 GPU runtime.
- Run the three cells (in order).
- Copy the printed out password/external IP from the second cell.
- Open the URL printed from the third cell (the one afer `your url is:`).
- Paste in the password/external IP and press "Click to Submit".

You should now be on the dashboard and able to use it fully, powered by a Google Colab GPU!

### Local

The dashboard can be deployed locally fairly easily. However, currently only CUDA and CPU PyTorch
devices are supported (i.e. Apple Metal and AMD ROCm are not). Furthermore, CPU essentially will
not work, with reconstructions both taking an absolute age and having strange artifacting (possibly
some translation issue from the pre-trained model). As such, to run this locally you realistically
_must_ have an NVidia GPU (with CUDA and PyTorch installed as required per your machine). You can
then follow the following simple steps:

- Use [`poetry`](https://python-poetry.org/docs/basic-usage/) to install the requisite dependencies.
- From this [link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing),
  download the checkpoint `ffhq_10m.pt` to the `/data/models/` directory.
- From project root, run `streamlit run app.py`.
- Open link printed out in the console.

You can also check out the
[original repo's Dockerfile](https://github.com/DPS2022/diffusion-posterior-sampling/blob/main/Dockerfile)
(potentially in conjuction with
[Streamlit's recommended deployment Dockerfile](https://docs.streamlit.io/deploy/tutorials/docker))
for alternative, containerized approaches, though these are untested!

## Future work

As mentioned, there are several potential avenues for improvement, with several tasks depending on
others. In roughly my order of priority, the following future work should be conducted:

- Add Pydantic models for each of config files in manner like for `DashboardConfig`.
- Add full diffusion sampler configurability (e.g. noise scheduling, posterior mean and variance types, noise clipping, etc.).
- Add full task configurability.
- Re-write lifted code from original repo.
- Re-add support for custom image uploading.
- Add full dataset loading support.
- Add better support for different models.
- Add ability to retrain models on the fly.
- Assess swapping to Jax.
- Assess Streamlit improvements (e.g. better caching, state management).
- Assess proper deployment options/strategy (e.g. using AWS with permanent site).
- Add more tests for other core modules, improving coverage.
