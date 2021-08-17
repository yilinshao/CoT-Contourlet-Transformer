import numpy as np
from tqdm import tqdm
import torch
from torch.nn.functional import conv2d
import time
import torch.nn as nn


def transform_filter(dir_filter):
    FColLength, FRowLength = dir_filter.shape
    new_filters = torch.zeros(2 * FColLength - 1, 2 * FRowLength - 1).float().cuda()
    index_mapping = torch.zeros(FColLength, FRowLength, 2).int().cuda()
    for i in range(FRowLength):
        index_mapping[:, i, 0] = torch.arange(new_filters.shape[0] - 1 - i, new_filters.shape[0] - 1 - i - FColLength, -1).int()
        index_mapping[:, i, 1] = torch.arange(FRowLength - 1 - i, FRowLength - 1 - i + FColLength).int()

    new_filters[index_mapping.reshape(-1, 2)[:, 0].long(), index_mapping.reshape(-1, 2)[:, 1].long()] = dir_filter.reshape(-1, )
    return new_filters




def zconv2(signal, dir_filter, mmatrix, gpu_mode=False):

    SColLength, SRowLength = signal.shape
    FColLength, FRowLength = dir_filter.shape

    FArray = dir_filter
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

    # index = np.zeros((FColLength * FRowLength * SColLength * SRowLength, 2)).astype(np.int)

    # filter_size = FRowLength * FColLength

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
                        sum += SArray[indexy, indexx] * FArray[l2, l1]
                        # index[(n1 * SColLength + n2) * filter_size + (l1 * FColLength + l2), 0] = indexy
                        # index[(n1 * SColLength + n2) * filter_size + (l1 * FColLength + l2), 1] = indexx
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

    else:
        signal = signal.unsqueeze(0).unsqueeze(0)
        dir_filter = transform_filter(dir_filter).unsqueeze(0).unsqueeze(0)

        pad = nn.ReflectionPad2d(padding=(FColLength - 1, FColLength - 1, FColLength - 1, FColLength - 1))
        signal = pad(signal)
        outArray = torch.nn.functional.conv2d(signal, dir_filter)
        outArray = outArray.squeeze(0).squeeze(0)
    return outArray
