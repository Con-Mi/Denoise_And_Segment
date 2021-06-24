from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torchvision
from skimage import io as sk_io
from skimage import color as sk_color
from skimage import util as sk_util
from typing import Optional


class PetsData(Dataset):
    def __init__(self, transform=None):
        r"""
        @brief              Constructor
        :param transform:
        """
        self._data_root: Path = Path("data")
        self._input_img: list = sorted(self._data_root.glob("**/*.jpg"))
        self._target_img: list = sorted(self._data_root.glob("**/*.png"))
        self._transform = transform

    def __len__(self):
        return len(self._input_img)

    def __getitem__(self, item):
        r"""
        @brief              Get Item
        :param item:
        :return:
        """
        inpt_img_path: Path = self._input_img[item]
        outpt_img_path: Path = self._target_img[item]

        inpt_img_str: str = inpt_img_path.as_posix()
        outpt_img_str: str = outpt_img_path.as_posix()

        inpt_img = sk_io.imread(inpt_img_str)
        print(inpt_img.shape)
        if len(inpt_img.shape) > 2:
            if inpt_img.shape[2] == 4:
                inpt_img = sk_color.rgba2rgb(inpt_img)
            inpt_noisy_img = sk_util.random_noise(inpt_img)
            target_img = sk_io.imread(outpt_img_str)
        else:
            inpt_img = sk_color.gray2rgb(inpt_img)
            inpt_noisy_img = sk_util.random_noise(inpt_img)
            target_img = sk_io.imread(outpt_img_str)

        if self._transform is not None:
            inpt_img = self._transform(inpt_img)
            inpt_noisy_img = self._transform(inpt_noisy_img)
            target_img = self._transform(target_img)

        return inpt_img, inpt_noisy_img, target_img


class PetsDataValid(Dataset):
    def __init__(self, transform=None):
        r"""
        @brief              Constructor
        :param transform:
        """
        self._data_root: Path = Path("data")
        self._input_img: list = sorted(self._data_root.glob("**/*.jpg"))[:100]
        self._target_img: list = sorted(self._data_root.glob("**/*.png"))[:100]
        self._transform = transform

    def __len__(self):
        return len(self._input_img)

    def __getitem__(self, item):
        r"""
        @brief              Get Item
        :param item:
        :return:
        """
        inpt_img_path: Path = self._input_img[item]
        outpt_img_path: Path = self._target_img[item]

        inpt_img_str: str = inpt_img_path.as_posix()
        outpt_img_str: str = outpt_img_path.as_posix()

        inpt_img = sk_io.imread(inpt_img_str)
        if inpt_img.shape[2] == 4:
            inpt_img = sk_color.rgba2rgb(inpt_img)
        inpt_noisy_img = sk_util.random_noise(inpt_img)
        target_img = sk_io.imread(outpt_img_str)

        if self._transform is not None:
            inpt_img = self._transform(inpt_img)
            inpt_noisy_img = self._transform(inpt_noisy_img)
            target_img = self._transform(target_img)

        return inpt_img, inpt_noisy_img, target_img


def PetsDataLoader(data_transform, BatchSz: int=2, worker_threads: int=2, shuffleOn: bool=True):
    if data_transform is None:
        data_transform = torchvision.transforms.ToTensor()
    ds = PetsData(transform=data_transform)

    pets_dataloader = DataLoader(ds, batch_size=BatchSz, num_workers=worker_threads, shuffle=shuffleOn)
    return pets_dataloader


def PetsValidationDataLoader(data_transform, BatchSz: int=2, worker_threads: int=2, shuffleOn: bool=True):
    if data_transform is None:
        data_transform = torchvision.transforms.ToTensor()

    ds = PetsDataValid(transform=data_transform)

    pets_dataloader = DataLoader(ds, batch_size=BatchSz, num_workers=worker_threads, shuffle=shuffleOn)
    return pets_dataloader

"""
from matplotlib import pyplot as plt
import numpy as np

ds = PetsDataLoader(None, 1, 2, True)

inpt = np.ndarray
noise_inpt = np.ndarray
segm = np.ndarray

for item in ds:
    inpt_img, noisy_img, segm_img = item
    inpt = inpt_img
    noise_inpt = noisy_img
    segm = segm_img
    to_plot = inpt_img
    if len(inpt_img.shape) != 3:
        break


#plt.imshow(to_plot.squeeze())
#plt.show()
"""