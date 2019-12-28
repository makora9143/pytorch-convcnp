import numpy as np


def cast_numpy(tensor):
    if not isinstance(tensor, np.ndarray):
        return tensor.numpy()
    return tensor


def channel_last(x):
    return x.transpose(1, 2).transpose(2, 3)


def channel_first(x):
    return x.transpose(3, 2).transpose(2, 1)
