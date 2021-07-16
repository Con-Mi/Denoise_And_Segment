import torch


def save_model(cust_model, name: str="dense_segm.pt"):
    r"""

    :param cust_model:
    :param name:
    :return:
    """
    return torch.save(cust_model.state_dict(), name)


def load_model(cust_model, model_dir: str="dense_segm.pt", map_location_device: str="cpu"):
    r"""

    :param cust_model:
    :param model_dir:
    :param map_location_device:
    :return:
    """
    if map_location_device == "cpu":
        cust_model.load_state_dict(torch.load(model_dir, map_location=map_location_device))
    elif map_location_device == "gpu":
        cust_model.load_state_dict(torch.load(model_dir))
    cust_model.eval()
    return cust_model