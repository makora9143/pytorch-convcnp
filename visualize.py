import PIL.Image
import io
import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

from utils import cast_numpy


def plot_model_output(xt, y_pred, std, color):
    y_pred = cast_numpy(y_pred).reshape(-1)
    xt = cast_numpy(xt[0]).reshape(-1)
    std = cast_numpy(std).reshape(-1)

    plt.plot(xt, y_pred, color=color)
    return plt.fill_between(xt,
                            y_pred - std,
                            y_pred + std,
                            color=color,
                            alpha=0.2)


def plot_all(xc, yc, xt, yt, model_xt, gp_pred, np_pred, support=False):
    gp_y_pred, gp_std = gp_pred
    plt.plot(xc.cpu()[0], yc.cpu()[0], 'o', color='black')
    plt.plot(xt.cpu()[0], yt.cpu()[0], '-', color='black')
    plot_model_output(model_xt.cpu(), gp_y_pred, gp_std, 'green')
    if support:
        plt.vlines([-2, 2], -3, 3, linestyles='dashed')
    plt.ylim(-3, 3)
    return plot_model_output(model_xt.cpu(), np_pred.mean.cpu()[0], np_pred.scale_tril.cpu()[0, :, 0, :], 'purple')


def convert_tfboard(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.clf()
    plt.close()
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    buf.close()
    return image


def show(img):
    npimg = img.cpu().numpy()
    return plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


def plot_all_2d(I, Mc, np_pred):
    img = make_grid(torch.cat([I.cpu(), Mc.cpu(), (Mc * I).cpu(), np_pred.cpu().clamp(0, 1)], 0))
    return show(img)
