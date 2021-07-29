from .efilter2 import *
from .zconv2 import *
from .zconv2S import *


def nssfbdec(x, f1, f2, nsdf_level, stage, mup=None, gpu_mode=False, index_dict=None):
    if mup is None:
        y1 = efilter2(x, f1)
        y2 = efilter2(x, f2)
        return y1, y2
    if mup is 1 or (np.asarray(mup) == np.eye(2)).all():
        y1 = efilter2(x, f1)
        y2 = efilter2(x, f2)
        return y1, y2
    if np.asarray(mup).shape == (2, 2):
        y1 = zconv2(x, f1, nsdf_level, stage, 1, mup, gpu_mode=gpu_mode, index_dict=index_dict)
        y2 = zconv2(x, f2, nsdf_level, stage, 2, mup, gpu_mode=gpu_mode, index_dict=index_dict)
    elif (np.asarray(mup).shape == np.asarray([1, 1])).all():
        y1 = zconv2S(x, f1, mup)
        y2 = zconv2S(x, f2, mup)
    else:
        raise ValueError('The upsampling parameter should be an integer or two-dimensional integer matrix!')

    return y1, y2