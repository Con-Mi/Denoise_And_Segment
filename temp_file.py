from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
from skimage import io as sk_io
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage import util as sk_util
import random


inputFiles: list = sorted(Path("data").glob("**/*.jpg"))
targetFiles: list = sorted(Path("data").glob("**/*.png"))

indx: int = random.randint(0, 5000)

print("INDEX IS : ", indx)

inpt_img_str = inputFiles[indx]

inpt_img = sk_io.imread(inpt_img_str)
inpt_noisy_img = sk_util.random_noise(inpt_img)
#target_img = sk_io.imread(outpt_img_str)

inptImg = Image.open(inputFiles[indx])
targtImg = Image.open(targetFiles[indx])

all_files: list = [inptImg, targtImg, inpt_noisy_img]

"""
fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(1, 3),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

for ax, im in zip(grid, [inptImg, targtImg, inpt_noisy_img]):
    # Iterating over the grid returns the Axes.
    ax.imshow(im)

plt.imshow(inpt_noisy_img)
plt.show()
"""


for img in all_files:
    plt.imshow(img)
    plt.show()
