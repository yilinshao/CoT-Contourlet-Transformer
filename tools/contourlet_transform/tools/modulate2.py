from numpy import *
from scipy import signal


def modulate2(x, type, center=array([[0, 0]])):
    # Size and origin
    if x.ndim > 1:
        s = array([x.shape])
    else:
        x = array([x])
        s = array(x.shape)

    o = floor(s / 2.0) + 1 + center
    n1 = arange(1, s[0][0]+1) - o[0][0]
    n2 = arange(1, s[0][1]+1) - o[0][1]
    if str.lower(type[0]) == 'r':
        m1 = (-1)**n1
        m1 = expand_dims(m1, axis=0)
        y = x * tile(m1.conj().T, s[0][1])

    elif str.lower(type[0]) == 'c':
        m2 = (-1)**n2
        m2 = expand_dims(m2, axis=0)
        y = x * tile(m2, (s[0][0], 1))

    elif str.lower(type[0]) == 'b':
        m1 = (-1)**n1
        m1 = expand_dims(m1, axis=0)
        m2 = (-1)**n2
        m2 = expand_dims(m2, axis=0)

        m = m1.conj().T * m2
        y = x * m

    return y

