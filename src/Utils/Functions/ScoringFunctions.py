import numpy as np
import torch
from sklearn.metrics import jaccard_score as sk_jaccard_score


def jaccard_score(y_true, y_pred):
    r"""

    :param y_true:
    :param y_pred:
    :return:
    """
    inputs = y_pred.view(-1)
    targets = y_true.view(-1)
    intersection = (targets * inputs).sum()
    total = (inputs + targets).sum()
    union = total - intersection
    IoU = (intersection + 1) / (union + 1)

    return IoU

def jaccard_sklearn_score(y_true: np.ndarray, y_pred: np.ndarray):
    r"""

    :param y_true:
    :param y_pred:
    :return:
    """
    return sk_jaccard_score(y_true, y_pred)

def dice_score(y_true, y_pred):
    r"""

    :param y_true:
    :param y_pred:
    :return:
    """
    #return (2.0 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)
    inputs = y_pred.view(-1)
    targets = y_true.view(-1)

    intersection = (inputs*targets).sum()
    dice = ( 2.0*intersection + 1.0 ) / (inputs.sum() + targets.sum() + 1.0)
    return dice


def mae_score(y_true, y_pred):
    r"""

    :param y_true:
    :param y_pred:
    :return:
    """
    return torch.mean( torch.abs(y_true - y_pred) )


def mse_score(y_true, y_pred):
    r"""

    :param y_true:
    :param y_pred:
    :return:
    """
    return torch.mean(torch.pow( input=(y_true-y_pred), exponent=2 ))


def psnr_score(y_true, y_hat):
    r"""

    :param y_true:
    :param y_hat:
    :return:
    """
    return (10 * torch.log10( 1/mse_score(y_true, y_hat) ))
