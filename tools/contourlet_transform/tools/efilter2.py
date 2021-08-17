import torch.nn.functional

from .extend2 import *
from scipy import signal

def efilter2(x, f, extmod='per', shift=None, gpu_mode=False):
    if shift is None:
        shift = [0, 0]
    sf = (np.asarray(f.shape) - 1) / 2

    xext = extend2(x, int(np.floor(sf[0])) + shift[0], int(np.ceil(sf[0])) - shift[0], int(np.floor(sf[1])) + shift[1],
                   int(np.ceil(sf[1])) - shift[1], extmod)
    if gpu_mode:
        y = torch.nn.functional.conv2d(xext.unsqueeze(0).unsqueeze(0), f.unsqueeze(0).unsqueeze(0))
        y = y.squeeze(0).squeeze(0)
    else:
        y = signal.convolve2d(xext, f, 'valid')
    return y
