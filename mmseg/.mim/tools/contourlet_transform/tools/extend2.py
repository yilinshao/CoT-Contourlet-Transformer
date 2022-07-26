import numpy as np


def extend2(x, ru, rd, cl, cr, extmod, gpu_mode):
    rx, cx = x.shape[-2:]

    if extmod == 'per':
        if gpu_mode:
            I = getPerIndices(rx, ru, rd)
            y = x[:, :, I, :]

            I = getPerIndices(cx, cl, cr)
            y = y[:, :, :, I]
        else:
            I = getPerIndices(rx, ru, rd)
            y = x[I, :]

            I = getPerIndices(cx, cl, cr)
            y = y[:, I]

    else:
        raise ValueError('Invalid input for EXTMOD')
    return y

def getPerIndices(lx, lb, le):
    if lx < lb:
        raise ValueError('lx < lb')
    I = np.concatenate((np.asarray(range(lx - lb, lx)), np.asarray(range(lx)), np.asarray(range(le))))
    return I