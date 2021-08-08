import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style


style.use("ggplot")

IMG_SIZE: int = 256
BATCH_SZ: int = 10

train_loss: list = [0.07579810, 0.03223921, 0.02740334, 0.02466767, 0.02374733, 0.02173243, 0.02000960, 0.01957438, 0.01887236, 0.01832653, 0.01794544, 0.01772916, 0.01694962, 0.01659123, 0.01610563, 0.01559546, 0.01612122, 0.01564613, 0.01500670, 0.01490935]
MAE: list        = [0.00918093, 0.00213359, 0.00237010, 0.00285651, 0.00243350, 0.00238057, 0.00224457, 0.00183417, 0.00195862, 0.00103887, 0.00181496, 0.00168101, 0.00143228, 0.00149646, 0.00131150, 0.00115973, 0.00105380, 0.00109190, 0.00094140, 0.00097666 ]
MSE:list         = [0.00119244, 0.00000804, 0.00000954, 0.00001349, 0.00001109, 0.00000976, 0.00000903, 0.00000752, 0.00000662, 0.00000555, 0.00000517, 0.00000414, 0.00000421, 0.00000340, 0.00000297, 0.00000318, 0.00000278, 0.00000244, 0.00000233, 0.00000200 ]
PSNR: list       = [19.62084961, 22.07952499, 22.74698448, 23.20913124, 23.38059044, 23.77310181, 24.10354614, 24.20515823, 24.36219215, 24.52068901, 24.58832741, 24.62990379, 24.84514809, 24.93244362, 25.06336021, 25.19392395, 25.07171822, 25.19159508, 25.34930801, 25.40096855 ]

valid_loss: list = [0.04293795, 0.03546762, 0.02448067, 0.02408483, 0.04577061, 0.03267045, 0.05115246, 0.01782356, 0.02149879, 0.01899982, 0.01667608, 0.01817201, 0.02318719, 0.01772509, 0.03915493, 0.02310520, 0.01567265, 0.02434730, 0.02020663, 0.01403019]
MAE_VALID: list  = [0.00228431, 0.00168100, 0.00252491, 0.00303103, 0.00128972, 0.00367260, 0.00350113, 0.00272651, 0.00297787, 0.00146544, 0.00136988, 0.00135354, 0.00184259, 0.00121598, 0.00762877, 0.00413091, 0.00026055, 0.00131666, 0.00019362, 0.00274151 ]
MSE_VALID:list   = [0.00000969, 0.00000489, 0.00000864, 0.00001158, 0.00000274, 0.00001612, 0.00001445, 0.00000915, 0.00001036, 0.00000260, 0.00000228, 0.00000255, 0.00000458, 0.00000195, 0.00006730, 0.00001973, 0.00000014, 0.00000214, 0.00000010, 0.00000887 ]
PSNR_VALID: list = [20.72920036, 21.54309273, 23.20295906, 23.24866295, 20.45528984, 21.91128731, 20.00444221, 24.61569786, 23.75366020, 24.31652069, 24.88123322, 24.49877930, 23.41728020, 24.59718323, 21.17535400, 23.44491386, 25.18359947, 23.24858284, 24.02128983, 25.63445091]


plt.plot(np.arange(len(train_loss)), PSNR, color='xkcd:magenta', marker="+")
plt.xlabel("Epoch Number")
plt.ylabel("Loss Value")
plt.plot(np.arange(len(valid_loss)), PSNR_VALID, color="xkcd:turquoise", marker="+")
plt.legend(["PSNR Loss Training", "PSNR Loss Validation"])
plt.title("Peak Signal To Noise - Training vs Validation")
plt.show()