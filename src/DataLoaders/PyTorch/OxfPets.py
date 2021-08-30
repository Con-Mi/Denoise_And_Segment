from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import torchvision
from skimage import io as sk_io
from skimage import color as sk_color
from skimage import util as sk_util


class PetsData(Dataset):
    def __init__(self, input_imgs: list, target_imgs: list, transform=None):
        r"""
        @brief              Constructor
        :param transform:
        """
        self._input_img: list = input_imgs
        self._target_img: list = target_imgs
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
        if len(inpt_img.shape) > 2:
            if inpt_img.shape[2] == 4:
                inpt_img = sk_color.rgba2rgb(inpt_img)
            inpt_noisy_img = sk_util.random_noise(inpt_img)
            target_img = sk_io.imread(outpt_img_str)
            target_img -= target_img
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
    def __init__(self, input_imgs: list, target_imgs: list, transform=None):
        r"""
        @brief              Constructor
        :param transform:
        """
        self._input_img: list = input_imgs
        self._target_img: list = target_imgs
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
        if len(inpt_img.shape) > 2:
            if inpt_img.shape[2] == 4:
                inpt_img = sk_color.rgba2rgb(inpt_img)
            inpt_noisy_img = sk_util.random_noise(inpt_img)
            target_img = sk_io.imread(outpt_img_str)
            target_img -= 1
        else:
            inpt_img = sk_color.gray2rgb(inpt_img)
            inpt_noisy_img = sk_util.random_noise(inpt_img)
            target_img = sk_io.imread(outpt_img_str)
            target_img -= 1

        if self._transform is not None:
            inpt_img = self._transform(inpt_img)
            inpt_noisy_img = self._transform(inpt_noisy_img)
            target_img = self._transform(target_img)

        return inpt_img, inpt_noisy_img, target_img


def train_validation_split(train_split_percentage: float=0.8):
    r"""

    :param train_split_percentage:
    :return:
    """
    dataRoot = Path("data")
    inputFiles: list = sorted(dataRoot.glob("**/*.jpg"))
    targetFiles: list = sorted(dataRoot.glob("**/*.png"))

    totalNrOfFiles: int = len(inputFiles)
    trainSize: float = train_split_percentage
    validSize: float = 1.0 - trainSize

    totalNrOfTrainD: int = round(totalNrOfFiles * trainSize)
    totalNrOfValidD: int = round(totalNrOfFiles * validSize)

    sampledValidIndxs: list = random.sample(range(len(inputFiles)), totalNrOfValidD)
    validInput: list = [inputFiles[x] for x in sampledValidIndxs]
    validTarget: list = [targetFiles[x] for x in sampledValidIndxs]

    trainInput: list = [inputFiles[x] for x in range(len(inputFiles)) if x not in sampledValidIndxs]
    trainTarget: list = [targetFiles[x] for x in range(len(inputFiles)) if x not in sampledValidIndxs]

    print("Length of Validation Input: ", len(validInput))
    print("Length of Validation Target: ", len(validTarget))
    print("Length of Training Input: ", len(trainInput))
    print("Length of Training Target: ", len(trainTarget))

    return trainInput, trainTarget, validInput, validTarget


def PetsDataLoader(data_transform, BatchSz: int=2, worker_threads: int=2, shuffleOn: bool=True):
    trainInputFiles, trainTargetFiles, _, _ = train_validation_split()
    if data_transform is None:
        data_transform = torchvision.transforms.ToTensor()
    ds = PetsData(input_imgs=trainInputFiles, target_imgs=trainTargetFiles, transform=data_transform)

    pets_dataloader = DataLoader(ds, batch_size=BatchSz, num_workers=worker_threads, shuffle=shuffleOn)
    return pets_dataloader


def PetsValidationDataLoader(data_transform, BatchSz: int=2, worker_threads: int=2, shuffleOn: bool=True):
    _, _, validInputFiles, validTargetFiles = train_validation_split()
    if data_transform is None:
        data_transform = torchvision.transforms.ToTensor()

    ds = PetsDataValid(input_imgs=validInputFiles, target_imgs=validTargetFiles, transform=data_transform)

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