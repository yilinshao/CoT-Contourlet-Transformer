import numpy as np
import torch

from tools.dfilters import dfilters
from math import sqrt
from tools.modulate2 import modulate2
from tools.parafilters import parafilters
from tools.atrousfilters import atrousfilters
from tools.nsfbdec import nsfbdec
from tools.nsdfbdec import nsdfbdec
import time


class ContourletDec:

    def __init__(self, levels, dfilter, pfilter, gpu=False):
        self.levels = levels
        self.dfilter = dfilter
        self.pfilter = pfilter
        self.gpu = gpu
        self.clevels = len(levels)
        self.nIndex = self.clevels + 1
        self.h1, self.h2, self.filters = self.get_filters()

    def nsctdec(self, x, gpu_mode=False):
        filters = []
        h1, h2 = dfilters(self.dfilter, 'd')
        h1 = h1 / sqrt(2)
        h2 = h2 / sqrt(2)
        filters.append(modulate2(h1, 'c'))
        filters.append(modulate2(h2, 'c'))
        h1, h2 = parafilters(h1, h2)
        filters.append(h1)
        filters.append(h2)
        if gpu_mode:

            level_012_nsdfb_level2_stage2_zconv1 = np.load('/data/level_012_nsdfb_level{}_stage{}_zconv{}.npy'.format(2, 2, 1))
            level_012_nsdfb_level2_stage2_zconv2 = np.load('/data/level_012_nsdfb_level{}_stage{}_zconv{}.npy'.format(2, 2, 2))
            level_012_nsdfb_level2_stage3_zconv1 = np.load('/data/level_012_nsdfb_level{}_stage{}_zconv{}.npy'.format(2, 3, 1))
            level_012_nsdfb_level2_stage3_zconv2 = np.load('/data/level_012_nsdfb_level{}_stage{}_zconv{}.npy'.format(2, 3, 2))

            level_012_nsfb_level0_atrous1 = np.load('/data/level_012_nsfb_level{}_atrous{}.npy'.format(0, 1))
            level_012_nsfb_level0_atrous2 = np.load('/data/level_012_nsfb_level{}_atrous{}.npy'.format(0, 2))
            level_012_nsfb_level1_atrous1 = np.load('/data/level_012_nsfb_level{}_atrous{}.npy'.format(1, 1))
            level_012_nsfb_level1_atrous2 = np.load('/data/level_012_nsfb_level{}_atrous{}.npy'.format(1, 2))

            index_dict = {'level_012_nsfb_level0_atrous1': level_012_nsfb_level0_atrous1,
                          'level_012_nsfb_level0_atrous2': level_012_nsfb_level0_atrous2,
                          'level_012_nsfb_level1_atrous1': level_012_nsfb_level1_atrous1,
                          'level_012_nsfb_level1_atrous2': level_012_nsfb_level1_atrous2,
                          'level_012_nsdfb_level2_stage2_zconv1': level_012_nsdfb_level2_stage2_zconv1,
                          'level_012_nsdfb_level2_stage2_zconv2': level_012_nsdfb_level2_stage2_zconv2,
                          'level_012_nsdfb_level2_stage3_zconv1': level_012_nsdfb_level2_stage3_zconv1,
                          'level_012_nsdfb_level2_stage3_zconv2': level_012_nsdfb_level2_stage3_zconv2}

        else:
            index_dict = None

        h1, h2, _, _ = atrousfilters(self.pfilter)

        clevels = len(self.levels)
        nIndex = clevels + 1
        y = [None for i in range(nIndex)]

        nsct_start_time = time.time()
        for i in range(1, clevels + 1):
            print('Start nsfb level {}'.format(self.levels[nIndex - 2]))
            # start_time = time.time()
            xlo, xhi = nsfbdec(x, h1, h2, i - 1, gpu_mode, nlevel=self.levels[nIndex - 2], index_dict=index_dict)
            # end_time = time.time()
            print('Finish nsfb level {}'.format(self.levels[nIndex - 2]))
            # print("Time:%.2fs" % (end_time - start_time))
            if self.levels[nIndex - 2] > 0:
                print('Start nsdfb level {}'.format(self.levels[nIndex - 2]))
                # start_time = time.time()
                xhi_dir = nsdfbdec(xhi, filters, self.levels[nIndex - 2], gpu_mode, index_dict)
                # end_time = time.time()
                print('Finish nsdfb level {}'.format(self.levels[nIndex - 2]))
                # print("Time:%.2fs" % (end_time - start_time))
                y[nIndex - 1] = xhi_dir
            else:
                y[nIndex - 1] = xhi
            nIndex = nIndex - 1
            x = np.copy(xlo)
        y[0] = x
        nsct_end_time = time.time()
        print("Time:%.2fs" % (nsct_end_time - nsct_start_time))
        return y

    def np_lst2torch_lst(self, filter_list):
        if isinstance(filter_list, list):
            new_list = []
            for filter_instance in filter_list:
                new_list.append(self.np_lst2torch_lst(filter_instance))
            return new_list
        else:
            return torch.from_numpy(filter_list).float().cuda()

    def get_filters(self):
        filters = []
        h1, h2 = dfilters(self.dfilter, 'd')
        h1 = h1 / sqrt(2)
        h2 = h2 / sqrt(2)
        filters.append(modulate2(h1, 'c'))
        filters.append(modulate2(h2, 'c'))
        h1, h2 = parafilters(h1, h2)
        filters.append(h1)
        filters.append(h2)
        h1, h2, _, _ = atrousfilters(self.pfilter)
        if self.gpu:
            h1 = torch.from_numpy(h1).cuda().float()
            h2 = torch.from_numpy(h2).cuda().float()
            filters = self.np_lst2torch_lst(filters)
        return h1, h2, filters

    def dec_iter(self, x):
        nIndex = self.nIndex
        y = [None for i in range(nIndex)]
        for i in range(1, self.clevels + 1):
            # print('Start nsfb level {}'.format(self.levels[nIndex - 2]))
            # start_time = time.time()
            xlo, xhi = nsfbdec(x, self.h1, self.h2, i - 1, gpu_mode=self.gpu, nlevel=self.levels[self.nIndex - 2])
            # end_time = time.time()
            # print('Finish nsfb level {}'.format(self.levels[nIndex - 2]))
            # print("Time:%.2fs" % (end_time - start_time))
            if self.levels[nIndex - 2] > 0:
                # print('Start nsdfb level {}'.format(self.levels[nIndex - 2]))
                # start_time = time.time()
                xhi_dir = nsdfbdec(xhi, self.filters, self.levels[nIndex - 2], gpu_mode=self.gpu)
                # end_time = time.time()
                # print('Finish nsdfb level {}'.format(self.levels[nIndex - 2]))
                # print("Time:%.2fs" % (end_time - start_time))
                y[nIndex - 1] = xhi_dir
            else:
                y[nIndex - 1] = xhi
            nIndex = nIndex - 1
            if self.gpu:
                x = torch.clone(xlo)
            else:
                x = np.copy(xlo)
        y[0] = x
        return y


