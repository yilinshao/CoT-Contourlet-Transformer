import numpy as np
from .symext import *
from .upsample2df import *
from scipy import signal
from .atrous import *


def nsfbdec(x, h0, h1, lev, gpu_mode=False, nlevel=None, index_dict=None):
    if lev != 0:
        l2 = np.eye(2)
        shift = (-2**(lev-1))*np.array([1, 1]) + 2
        L = 2**lev

        y0 = upsample2df(h0, lev)
        y0 = symext(x, y0, shift)
        y0 = atrous(y0, h0, l2 * L, nlevel=nlevel, gpu_mode=gpu_mode, num=1, index_dict=index_dict)

        y1 = upsample2df(h1, lev)
        y1 = symext(x, y1, shift)
        y1 = atrous(y1, h1, l2 * L, nlevel=nlevel, gpu_mode=gpu_mode, num=2, index_dict=index_dict)
    else:
        shift = [1, 1]
        y0 = symext(x, h0, shift)
        y0 = signal.convolve2d(y0, h0, 'valid')
        y1 = symext(x, h1, shift)
        y1 = signal.convolve2d(y1, h1, 'valid')
    return y0, y1
