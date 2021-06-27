import torch
from matplotlib import pyplot as plt
import torchvision
from pathlib import Path
from skimage import io
from skimage import color as sk_color
from skimage import util as sk_util
from src.Models.DenseLinkNet import DenseDenoisSegmentModel
from torchvision.utils import make_grid
from torchvision import io as tv_io
import torchvision.transforms.functional as F
import piq


def read_preprocess_img(file: str) -> tuple:
    img = io.imread(file)
    noisy_img = sk_util.random_noise(img)
    MUL_TRANFORMS: list = [ torchvision.transforms.ToTensor(), torchvision.transforms.Resize(size=(IMG_SIZE, IMG_SIZE)) ]
    newImg = MUL_TRANFORMS[0](img)
    newImg = MUL_TRANFORMS[1](newImg)
    newNoisyImg = MUL_TRANFORMS[0](noisy_img)
    newNoisyImg = MUL_TRANFORMS[1](newNoisyImg)
    return newImg, newNoisyImg


def main():
    files: list = sorted(Path("data").glob("**/*.jpg"))
    input, inputNoisy = read_preprocess_img(files[0])
    model = DenseDenoisSegmentModel()
    denoise_out, segm_out = model(input.unsqueeze(dim=0))
    ssim_index = piq.ssim(inputNoisy.unsqueeze(dim=0), input.unsqueeze(dim=0))
    print(ssim_index)

    denoise_out = denoise_out.squeeze()
    segm_out = torch.round(segm_out).squeeze()
    denoise_out = F.to_pil_image(denoise_out)
    segm_out = F.to_pil_image(segm_out)
    input = F.to_pil_image(input)

    plt.imshow(input)
    plt.show()


if __name__ == '__main__':

    IMG_SIZE: int = 384

    main()
