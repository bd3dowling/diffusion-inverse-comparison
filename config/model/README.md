# config.model

This is where all model configuration YAMLs live. These are the configs that were used for training
the models associated with the available checkpoints and, as such, should not be changed.
The directory is setup as a python package (module) so that the YAMLs can be loaded with
`importlib.resources.files`. The files are as follows:

- [`ffhq.yaml`](./ffhq.yaml): Config for the FFHQ pre-trained model.
- [`imagenet.yaml`](./imagenet.yaml): Config for the Imagenet pre-trained model.
