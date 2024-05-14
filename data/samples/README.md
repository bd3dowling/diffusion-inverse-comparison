# samples

This is where FFHQ image samples for use in the dashboard live. The current dashboard logic reads
in all of these files, so the addition or removal of more FFHQ samples will be reflected immediately
in the dashboard. The directory setup as a python package so that the checkpoint files can be loaded
with `importlib.resources.files`.
