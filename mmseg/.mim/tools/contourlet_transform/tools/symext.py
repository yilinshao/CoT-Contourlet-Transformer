import numpy as np
import torch
import torchvision.transforms.functional as TF


def symext(x, h, shift, gpu_mode=False):
    m, n = x.shape[-2:]

    p, q = h.shape[-2:]

    p2 = int(np.floor(p/2))
    q2 = int(np.floor(q/2))

    s1 = shift[0]
    s2 = shift[1]

    ss = p2 - s1 + 1
    rr = q2 - s2 + 1

    if gpu_mode:

        num_repeat_h = max((p + s1) // n, ss // n)
        n_exp = n * (num_repeat_h + 1)
        x_flip_cat_h = torch.cat((x, TF.hflip(x)), dim=3)
        x_expand_h = torch.cat((x_flip_cat_h.repeat(1, 1, 1, num_repeat_h), x), dim=3)

        yT = torch.cat((TF.hflip(x_expand_h[:, :, :, 0: ss]), x, TF.hflip(x_expand_h[:, :, :, n_exp - p - s1: n_exp])), dim=3)

        num_repeat_v = max((q + s2) // m, rr // m)
        m_exp = m * (num_repeat_v + 1)
        yT_flip_cat_v = torch.cat((yT, TF.vflip(yT)), dim=2)
        yT_expand_v = torch.cat((yT_flip_cat_v.repeat(1, 1, num_repeat_v, 1), yT), dim=2)

        yT = torch.cat((TF.vflip(yT_expand_v[:, :, 0: rr]), yT, TF.vflip(yT_expand_v[:, :, m_exp - q - s2: m_exp])), dim=2)
        yT = yT[:, :, 0: m + p - 1, 0: n + q - 1]

    else:
        yT = np.concatenate((np.fliplr(x[:, 0: ss]), x, x[:, n - 1: n - p - s1 - 1: -1]), axis=1)
        yT = np.concatenate((np.flipud(yT[0:rr, :]), yT, yT[m - 1: m - q - s2 - 1: -1]), axis=0)
        yT = yT[0: m + p - 1, 0: n + q - 1]
    return yT


