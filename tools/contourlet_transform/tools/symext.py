import numpy as np
import torch
import torchvision.transforms.functional as TF


def symext(x, h, shift, gpu_mode=False):
    m = x.shape[0]
    n = x.shape[1]

    p = h.shape[0]
    q = h.shape[1]

    p2 = int(np.floor(p/2))
    q2 = int(np.floor(q/2))

    s1 = shift[0]
    s2 = shift[1]

    ss = p2 - s1 + 1
    rr = q2 - s2 + 1

    if gpu_mode:
        yT = torch.cat((TF.hflip(x[:, 0: ss]), x, TF.hflip(x[:, n - p - s1: n])), dim=1)
        yTT = yT.cpu().numpy()
        yT = torch.cat((TF.vflip(yT[0: rr]), yT, TF.vflip(yT[m - q - s2: m])), dim=0)
    else:
        yT = np.concatenate((np.fliplr(x[:, 0: ss]), x, x[:, n - 1: n - p - s1 - 1: -1]), axis=1)
        yT = np.concatenate((np.flipud(yT[0:rr, :]), yT, yT[m - 1: m - q - s2 - 1: -1]), axis=0)
    yT = yT[0: m + p - 1, 0: n + q - 1]
    return yT


