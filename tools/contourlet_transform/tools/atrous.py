import numpy as np
from tqdm import tqdm
import torch
from torch.nn.functional import conv2d
import torchvision.transforms.functional as TF


def transform_filter(p_filter, m0, m3):

    p_filter = TF.rotate(p_filter, 180.0)
    p_filter_new = torch.zeros(p_filter.shape[2] * m3, p_filter.shape[3] * m0).cuda()
    p_filter_new[m3-1::m3, m0-1::m0] = p_filter[0, 0]

    return p_filter_new


def atrous(signal, filter, mmatrix, gpu_mode=False):
    SColLength = signal.shape[0]
    SRowLength = signal.shape[1]
    FColLength = filter.shape[0]
    FRowLength = filter.shape[1]

    SFColLength = FColLength - 1
    SFRowLength = FRowLength - 1
    FArray = filter
    SArray = signal
    M = mmatrix
    M0 = int(M.flatten('F')[0])
    M3 = int(M.flatten('F')[3])
    sM0 = M0 - 1
    sM3 = M3 - 1

    O_SColLength = SColLength - M0 * FColLength + 1
    O_SRowLength = SRowLength - M3 * FRowLength + 1

    if not gpu_mode:
        outArray = np.zeros((O_SColLength, O_SRowLength))
        index = np.zeros((FColLength * FRowLength * O_SColLength * O_SRowLength, 2)).astype(np.int)

        for n1 in tqdm(range(O_SRowLength)):
            for n2 in range(O_SColLength):
                sum = 0
                kk1 = n1 + sM0
                print('Fshape:{}, Sshape:{}'.format(FArray.shape, SArray.shape))
                for k1 in range(FRowLength):
                    kk2 = n2 + sM3
                    for k2 in range(FColLength):
                        f1 = SFRowLength - k1
                        f2 = SFColLength - k2
                        sum += FArray[f2, f1] * SArray[kk2, kk1]
                        index[(n1 * O_SColLength + n2) * FRowLength * FColLength + (k1 * FColLength + k2), 0] = kk2
                        index[(n1 * O_SColLength + n2) * FRowLength * FColLength + (k1 * FColLength + k2), 1] = kk1
                        print('F:[{},{}],S:[{},{}]'.format(f2, f1, kk2, kk1))
                        kk2 += M3
                    print('-' * 20)
                    kk1 += M0
                outArray[n2, n1] = sum
                # print('*'*20)
    else:
        signal = signal.unsqueeze(0).unsqueeze(0)
        p_filter = transform_filter(filter.unsqueeze(0).unsqueeze(0), M0, M3).unsqueeze(0).unsqueeze(0)
        outArray = conv2d(signal.float(), p_filter).squeeze(0).squeeze(0)

    return outArray
