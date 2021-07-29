from .nssfbdec import *
from tqdm import tqdm

def nsdfbdec(x, dfilter, clevels, gpu_mode=False, index_dict=None):
    k1 = dfilter[0]
    k2 = dfilter[1]

    f1 = dfilter[2]
    f2 = dfilter[3]

    q1 = [[1, -1], [1, 1]]
    y = []
    if clevels == 1:
        y1, y2 = nssfbdec(x, k1, k2, clevels, stage=1)
        y.append(y1)
        y.append(y2)
    else:
        x1, x2 = nssfbdec(x, k1, k2, clevels, stage=1)
        y1, y2 = nssfbdec(x1, k1, k2, clevels, stage=2, mup=q1, gpu_mode=gpu_mode, index_dict=index_dict)
        y3, y4 = nssfbdec(x2, k1, k2, clevels, stage=2, mup=q1, gpu_mode=gpu_mode, index_dict=index_dict)
        y.append(y1)
        y.append(y2)
        y.append(y3)
        y.append(y4)

        for l in tqdm(range(3, clevels + 1), desc='levels'):
            y_old = y
            y = [None for i in range(2**l)]
            for k in range(1, 2**(l-2) + 1):
                slk = 2 * int(np.floor((k-1)/2)) - 2**(l-3) + 1
                mkl = 2 * np.dot(np.asarray([[2**(l - 3), 0], [0, 1]]), np.asarray([[1, 0], [-slk, 1]]))
                i = np.remainder(k - 1, 2) + 1
                y[2*k-2], y[2*k-1] = nssfbdec(y_old[k-1], f1[i-1], f2[i-1], mkl)
            for k in range(2**(l-2) + 1, 2**(l-1) + 1):
                slk = 2 * int(np.floor((k-2**(l-2)-1) / 2)) - 2**(l-3) + 1
                mkl = 2 * np.dot(np.asarray([1, 0], [0, 2 ** (l - 3)]), np.asarray([[1, -slk], [0, 1]]))
                i = np.remainder(k - 1, 2) + 3
                y[2 * k - 2], y[2 * k - 1] = nssfbdec(y_old[k - 1], f1[i - 1], f2[i - 1], mkl)
    return y




