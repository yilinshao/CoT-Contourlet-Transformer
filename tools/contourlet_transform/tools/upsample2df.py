import numpy as np


def upsample2df(h, power):
    m, n = h.shape[0], h.shape[1]
    ho = np.zeros([2**power*m, 2**power*n])
    ho[::2**power, ::2**power] = h
    return ho
