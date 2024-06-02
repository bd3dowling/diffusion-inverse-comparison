"""Dataset definitions and registry."""

from importlib.resources import files
from pathlib import Path
from typing import Callable, Optional

import torchvision.transforms as transforms
from PIL import Image
from strenum import StrEnum
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from data.samples import ffhq, imagenet

__DATASET__ = {}


class DatasetType(StrEnum):
    FFHQ = "ffhq"
    IMAGENET = "imagenet"
    OWN = "own"


def register_dataset(name: DatasetType):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls

    return wrapper


def get_dataset(name: DatasetType, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](**kwargs)


def get_dataloader(dataset: VisionDataset, batch_size: int, num_workers: int, train: bool):
    dataloader = DataLoader(
        dataset, batch_size, shuffle=train, num_workers=num_workers, drop_last=train
    )
    return dataloader


@register_dataset(name=DatasetType.FFHQ)
class FFHQDataset(VisionDataset):
    def __init__(self, transforms: Optional[Callable] = None):
        root = Path(str(files(ffhq)))
        self.file_paths = [*root.glob("*.png")]

        super().__init__(str(root), transforms)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index: int):
        file_path = self.file_paths[index]
        img = Image.open(file_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        return img


@register_dataset(name=DatasetType.IMAGENET)
class ImageNetDataset(VisionDataset):
    def __init__(self, transforms: Optional[Callable] = None):
        root = Path(str(files(imagenet)))
        self.file_paths = [*root.glob("*.jpg")]

        super().__init__(str(root), transforms)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index: int):
        file_path = self.file_paths[index]
        img = Image.open(file_path).convert("RGB")

        if self.transforms is not None:
            img = transforms.Resize((256, 256))(img)
            img = self.transforms(img)

        return img
