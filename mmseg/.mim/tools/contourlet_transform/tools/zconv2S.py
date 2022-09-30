import numpy as np


def zconv2S(signal, filter, mmatrix):

    SColLength, SRowLength = signal.shape
    FColLength, FRowLength = filter.shape

    FArray = filter
    SArray = signal
    M = np.asarray(mmatrix)

    M0 = int(M.flatten('F')[0])
    M1 = int(M.flatten('F')[1])
    M2 = int(M.flatten('F')[2])
    M3 = int(M.flatten('F')[3])

    NewFRowLength = int((M.flatten('F')[0] - 1) * (FRowLength - 1)) + int(
        (M.flatten('F')[2]) * (FColLength - 1)) + FRowLength - 1
    NewFColLength = int((M.flatten('F')[3] - 1) * (FColLength - 1)) + int(
        (M.flatten('F')[1]) * (FRowLength - 1)) + FColLength - 1


    outArray = np.zeros((SColLength, SRowLength))

    sum = 0
    Start1 = int(NewFRowLength / 2)
    Start2 = int(NewFColLength / 2)
    mn1 = Start1 % SRowLength
    mn2 = mn2save = Start2 % SColLength

    for n1 in range(SRowLength):
        for n2 in range(SColLength):
            outindexx = mn1
            outindexy = mn2
            for l1 in range(FRowLength):
                indexx = outindexx
                indexy = outindexy
                for l2 in range(FColLength):
                    sum += SArray[indexy, indexx]*FArray[l2, l1]
                    # indexx -= M2
                    # if indexx < 0:
                    #     indexx += SRowLength
                    # if indexx > SRowLength - 1:
                    #     indexx -= SRowLength
                    # indexy -= M3

                    if indexy < 0:
                        indexy += SColLength
                outindexx -= M0
                if outindexx < 0:
                    outindexx += SRowLength
                # outindexy -= M1
                # if outindexy < 0:
                #     outindexy += SColLength
                # if outindexy > SColLength-1:
                #     outindexy -= SColLength
            outArray[n2, n1] = sum
            sum = 0
            mn2 += 1
            if mn2 > SColLength - 1:
                mn2 -= SColLength
        mn2 = mn2save
        mn1 += 1
        if mn1 > SRowLength - 1:
            mn1 -= SRowLength
    return outArray
