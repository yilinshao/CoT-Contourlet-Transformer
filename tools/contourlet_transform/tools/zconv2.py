import numpy as np
from tqdm import tqdm
import torch
from torch.nn.functional import conv2d
import time


def zconv2(signal, filter, nsdf_level, stage, num, mmatrix, gpu_mode=False, index_dict=None):

    SColLength, SRowLength = signal.shape
    FColLength, FRowLength = filter.shape

    FArray = filter
    SArray = signal
    M = np.asarray(mmatrix)

    M0 = int(M.flatten('F')[0])
    M1 = int(M.flatten('F')[1])
    M2 = int(M.flatten('F')[2])
    M3 = int(M.flatten('F')[3])

    NewFRowLength = int((M.flatten('F')[0] - 1) * (FRowLength - 1)) + int((M.flatten('F')[2]) * (FColLength - 1)) + FRowLength - 1
    NewFColLength = int((M.flatten('F')[3] - 1) * (FColLength - 1)) + int((M.flatten('F')[1]) * (FRowLength - 1)) + FColLength - 1

    outArray = np.zeros((SColLength, SRowLength))

    sum = 0
    Start1 = int(NewFRowLength / 2)
    Start2 = int(NewFColLength / 2)
    mn1 = Start1 % SRowLength
    mn2 = mn2save = Start2 % SColLength

    index = np.zeros((FColLength * FRowLength * SColLength * SRowLength, 2)).astype(np.int)

    filter_size = FRowLength * FColLength

    if gpu_mode is not True:
        for n1 in tqdm(range(SRowLength), desc='n1'):
            for n2 in range(SColLength):
                outindexx = mn1
                outindexy = mn2
                # print('Sshape:{}, Fshape:{}'.format(SArray.shape, FArray.shape))

                for l1 in range(FRowLength):
                    indexx = outindexx
                    indexy = outindexy
                    for l2 in range(FColLength):
                        sum += SArray[indexy, indexx]*FArray[l2, l1]
                        index[(n1 * SColLength + n2) * filter_size + (l1 * FColLength + l2), 0] = indexy
                        index[(n1 * SColLength + n2) * filter_size + (l1 * FColLength + l2), 1] = indexx
                        # sum = 0
                        # print('S:[{}, {}], F[{}, {}]'.format(indexy, indexx, l2, l1))
                        indexx -= M2
                        if indexx < 0:
                            indexx += SRowLength
                        if indexx > SRowLength - 1:
                            indexx -= SRowLength
                        indexy -= M3
                        if indexy < 0:
                            indexy += SColLength
                    # print('-'*20)
                    outindexx -= M0
                    if outindexx < 0:
                        outindexx += SRowLength
                    outindexy -= M1
                    if outindexy < 0:
                        outindexy += SColLength
                    if outindexy > SColLength-1:
                        outindexy -= SColLength
                outArray[n2, n1] = sum
                # print('*' * 20)
                sum = 0
                mn2 += 1
                if mn2 > SColLength - 1:
                    mn2 -= SColLength
            mn2 = mn2save
            mn1 += 1
            if mn1 > SRowLength - 1:
                mn1 -= SRowLength
        save_npy_name = '/data/level_012_nsdfb_level{}_stage{}_zconv{}.npy'.format(nsdf_level, stage, num)

        np.save(save_npy_name, index)
        print('Wrote index to /data/level_012_nsdfb_level{}_stage{}_zconv{}.npy'.format(nsdf_level, stage, num))

    else:
        load_npy_name = 'level_012_nsdfb_level{}_stage{}_zconv{}'.format(nsdf_level, stage, num)

        index = torch.from_numpy(index_dict[load_npy_name]).cuda()
        SArray_cuda = torch.from_numpy(SArray).cuda()
        grid_image = SArray_cuda[index[:, 0], index[:, 1]].reshape((SRowLength, SColLength, FRowLength, FColLength))

        grid_image = grid_image.permute((0, 2, 1, 3)).reshape((SRowLength*FRowLength, SColLength*FColLength))

        grid_image = grid_image.unsqueeze(0).unsqueeze(0)

        FArray_cuda = torch.from_numpy(FArray[:, ::-1].copy()).unsqueeze(0).unsqueeze(0).cuda()

        outArray = conv2d(grid_image, FArray_cuda, stride=FColLength)

        outArray = outArray.squeeze(0).squeeze(0)
        outArray = outArray.cpu().numpy().transpose(1, 0)


    return outArray
