import numpy as np
from tqdm import tqdm
import torch
from torch.nn.functional import conv2d


def atrous(signal, filter, mmatrix, nlevel, gpu_mode=False, num=None, index_dict=None):
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
                # print('Fshape:{}, Sshape:{}'.format(FArray.shape, SArray.shape))
                for k1 in range(FRowLength):
                    kk2 = n2 + sM3
                    for k2 in range(FColLength):
                        f1 = SFRowLength - k1
                        f2 = SFColLength - k2
                        sum += FArray[f2, f1] * SArray[kk2, kk1]
                        index[(n1 * O_SColLength + n2) * FRowLength * FColLength + (k1 * FColLength + k2), 0] = kk2
                        index[(n1 * O_SColLength + n2) * FRowLength * FColLength + (k1 * FColLength + k2), 1] = kk1
                        # print('F:[{},{}],S:[{},{}]'.format(f2, f1, kk2, kk1))
                        kk2 += M3
                    # print('-' * 20)
                    kk1 += M0
                outArray[n2, n1] = sum
                # print('*'*20)
        save_npy_name = '/data/level_012_nsfb_level{}_atrous{}.npy'.format(nlevel, num)
        np.save(save_npy_name, index)
        print('Wrote index to /data/level_012_nsfb_level{}_atrous{}.npy'.format(nlevel, num))
    else:
        load_npy_name = 'level_012_nsfb_level{}_atrous{}'.format(nlevel, num)
        index = index_dict[load_npy_name]

        grid_image = SArray[index[:, 0], index[:, 1]].reshape((O_SRowLength, O_SColLength, FRowLength, FColLength))
        grid_image = grid_image.transpose((0, 2, 1, 3)).reshape((O_SRowLength * FRowLength, O_SColLength * FColLength))

        grid_image = torch.from_numpy(grid_image).unsqueeze(0).unsqueeze(0)
        FArray_cuda = torch.from_numpy(FArray[:, ::-1].copy()).unsqueeze(0).unsqueeze(0)

        outArray = conv2d(grid_image, FArray_cuda, stride=FColLength)
        outArray = outArray.squeeze(0).squeeze(0)
        outArray = outArray.numpy().transpose(1, 0)
    return outArray
