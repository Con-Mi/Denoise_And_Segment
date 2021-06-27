import torch
from sklearn.metrics import jaccard_score as sk_jaccard_score


def jaccard_score(y_true, y_pred):
    r"""

    :param y_true:
    :param y_pred:
    :return:
    """
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def scikit_jaccard_score(y_true, y_pred):
    sk_jaccard_score(y_true=y_true, y_pred=y_pred)


def dice_score(y_true, y_pred):
    r"""

    :param y_true:
    :param y_pred:
    :return:
    """
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


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


def psnr(y_true, y_hat):
    r"""

    :param y_true:
    :param y_hat:
    :return:
    """
    return (10 * torch.log10( 1/mse_score(y_true, y_hat) ))