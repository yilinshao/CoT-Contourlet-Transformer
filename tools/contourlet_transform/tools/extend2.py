import numpy as np


def extend2(x, ru, rd, cl, cr, extmod):
    rx, cx = x.shape

    if extmod == 'per':
        I = getPerIndices(rx, ru, rd)
        y = x[I, :]

        I = getPerIndices(cx, cl, cr)
        y = y[:, I]

    else:
        raise ValueError('Invalid input for EXTMOD')
    return y

def getPerIndices(lx, lb, le):
    I = np.concatenate((np.asarray(range(lx - lb, lx)), np.asarray(range(lx)), np.asarray(range(le))))
    return I