from .modulate2 import *
from .resampz import *

def parafilters(f1, f2):
    y1 = []
    y2 = []

    y1.append(modulate2(f1, 'r'))
    y2.append(modulate2(f2, 'r'))
    y1.append(modulate2(f1, 'c'))
    y2.append(modulate2(f2, 'c'))

    y1.append(y1[0].T)
    y2.append(y2[0].T)
    y1.append(y1[1].T)
    y2.append(y2[1].T)

    for i in range(4):
        y1[i] = resampz(y1[i], i)
        y2[i] = resampz(y2[i], i)

    return y1, y2
