def jaccard_score(y_true, y_pred):
    r"""

    :param y_true:
    :param y_pred:
    :return:
    """
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def dice_score(y_true, y_pred):
    r"""

    :param y_true:
    :param y_pred:
    :return:
    """
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

