import numpy as np
import torch


def upsample2df(h, power, gpu_mode):
    m, n = h.shape[0], h.shape[1]
    if gpu_mode:
        ho = torch.zeros([2**power*m, 2**power*n]).float().cuda()
        ho[::2 ** power, ::2 ** power] = h
    else:
        ho = np.zeros([2**power*m, 2**power*n])
        ho[::2**power, ::2**power] = h
    return ho
