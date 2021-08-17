import numpy as np
import torch.nn.functional

from .symext import *
from .upsample2df import *
from scipy import signal
from .atrous import *


def nsfbdec(x, h0, h1, lev, gpu_mode=False, nlevel=None):
    if lev != 0:
        l2 = np.eye(2)
        shift = (-2**(lev-1))*np.array([1, 1]) + 2
        L = 2**lev

        y0 = upsample2df(h0, lev, gpu_mode=gpu_mode)
        y0 = symext(x, y0, shift, gpu_mode=gpu_mode)
        y0 = atrous(y0, h0, l2 * L, gpu_mode=gpu_mode)

        y1 = upsample2df(h1, lev, gpu_mode=gpu_mode)
        y1 = symext(x, y1, shift, gpu_mode=gpu_mode)
        y1 = atrous(y1, h1, l2 * L, gpu_mode=gpu_mode)
    else:
        shift = [1, 1]
        y0 = symext(x, h0, shift, gpu_mode=gpu_mode)
        if gpu_mode:
            y0 = torch.nn.functional.conv2d(y0.unsqueeze(0).unsqueeze(0), h0.unsqueeze(0).unsqueeze(0))
            y0 = y0.squeeze(0).squeeze(0)
        else:
            y0 = signal.convolve2d(y0, h0, 'valid')

        y1 = symext(x, h1, shift, gpu_mode=gpu_mode)
        if gpu_mode:
            y1 = torch.nn.functional.conv2d(y1.unsqueeze(0).unsqueeze(0), h1.unsqueeze(0).unsqueeze(0))
            y1 = y1.squeeze(0).squeeze(0)
        else:
            y1 = signal.convolve2d(y1, h1, 'valid')
    return y0, y1
