import torch
from pathlib import Path
import random
import torchvision
from skimage import io as sk_io
from skimage import color as sk_color
from skimage import util as sk_util
from skimage.transform import resize as sk_resize
import numpy as np
from PIL import Image

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from src.Models.DenseLinkNet import DenseDenoisSegmentModel
from src.Utils.ModelSaving.ModelSavingFunctions import load_model


inputFiles: list = sorted(Path("data").glob("**/*.jpg"))
targetFiles: list = sorted(Path("data").glob("**/*.png"))

IMG_SIZE: int = 256

indx: int = random.randint(6800, 7300)

print("INDEX IS : ", indx)

SegmModel = DenseDenoisSegmentModel()

model = load_model(SegmModel, model_dir="saved_models/dense_linknet_20.pt")

inpt_img_str = inputFiles[indx]
targt_img_str = targetFiles[indx]

inpt_pil_img = Image.open(inpt_img_str).resize((IMG_SIZE, IMG_SIZE))
targt_pil_img = Image.open(targt_img_str).resize((IMG_SIZE, IMG_SIZE))

inpt_img = sk_io.imread(inpt_img_str)
targt_img = sk_io.imread(targt_img_str)
inpt_noisy_img = sk_util.random_noise(inpt_img)

inpt_noisy_img_resize = sk_resize(inpt_noisy_img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
inpt_img_resize = sk_resize( inpt_img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True )

MUL_TRANFORMS: list = torchvision.transforms.Compose( [ torchvision.transforms.ToTensor(), torchvision.transforms.Resize(size=(IMG_SIZE, IMG_SIZE)) ] )

inpt = MUL_TRANFORMS(inpt_noisy_img)
#inpt = MUL_TRANFORMS(inpt_img)

inptU = inpt.unsqueeze(dim=0)
inptU = inptU.to(device="cpu", dtype=torch.float)
denoised, segm = model(inptU)

fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(1, 4),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

output = segm.squeeze().detach().numpy()
denoised_output = denoised.squeeze().detach()
denoised_output = denoised_output.permute(1, 2, 0).numpy()
denoised_output = (denoised_output * 255).astype(np.uint8)

#img_pil = torchvision.transforms.ToPILImage()(denoised.squeeze().detach()).convert("RGB")
#targt_img_pil = torchvision.transforms.ToPILImage()(targt_img).convert("L")

#im.show()

for ax, im, title in zip(grid, [inpt_noisy_img_resize, np.asarray(targt_pil_img), denoised_output, output], ["Noisy Image Input", "Segmentation - True Label", "Denoised Prediction", "Segmentation Prediction"]):
    # Iterating over the grid returns the Axes.
    ax.set_title(title)
    ax.imshow(im)

plt.show()
