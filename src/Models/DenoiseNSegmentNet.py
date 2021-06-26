import torch
from DenseLinkNet import DenseSegmModel


class DenoiseSegmModel(torch.nn.Module):
    def __init__(self):
        super(DenoiseSegmModel).__init__()

        self._DenoiseNet = DenseSegmModel(input_channels=3, num_filters=32, num_classes=3, pretrained=True)
        self._SegmNet = DenseSegmModel(input_channels=3, num_filters=32, num_classes=3, pretrained=True)

    def forward(self, input):
        denoise_out = self._DenoiseNet(input)
        segm_out = self._SegmNet(denoise_out)

        return denoise_out, segm_out


def DenseDenoisSegmentModel():
    return DenoiseSegmModel()
