from torch.utils import data
import torch
import cv2
from contourlet_dec import ContourletDec
import os
import numpy as np
from PIL import Image
from tools.shownsct import shownsct, save_nsct, load_nsct, torch_lst2np_lst, norm_features
from tqdm import tqdm
import time


class CityscapesImages(data.Dataset):
    def __init__(self, root, list_path=None):
        self.root = root
        self.img_list = []
        city_dirs = os.listdir(root)
        for city in city_dirs:
            imgs_in_city = os.listdir(os.path.join(root, city))
            imgs_in_city = [os.path.join(city, img_in_city) for img_in_city in imgs_in_city]
            self.img_list.extend(imgs_in_city)

        # self.list_path = list_path
        #
        # self.img_list = [line.strip().split()[0] for line in open(os.path.join(root, list_path))]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.img_list[index])).convert('L')
        # img = img.resize((512, 256))
        img = np.array(img)
        img_path = os.path.join(self.root, self.img_list[index])
        return img.copy(), img_path


class PascalContext(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.img_list = os.listdir(root)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.img_list[index])).convert('L')
        # img = Image.open('../../data/VOCdevkit/VOC2010/JPEGImages/2008_001823.jpg').convert('L')

        # img = img.resize((512, 256))
        img = np.array(img)
        img_path = os.path.join(self.root, self.img_list[index])
        return img.copy(), img_path


class Ade(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.img_list = os.listdir(root)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.img_list[index])).convert('L')
        # img = Image.open('../../data/VOCdevkit/VOC2010/JPEGImages/2008_001823.jpg').convert('L')

        # img = img.resize((512, 256))
        img = np.array(img)
        img_path = os.path.join(self.root, self.img_list[index])
        return img.copy(), img_path


def main():
    # get filters
    levels = [0, 1]
    pfilter = 'maxflat'
    dfilter = 'dmaxflat7'
    nsct = ContourletDec(levels, dfilter, pfilter, gpu=True)

    # get dataloader
    dec_dataset = CityscapesImages('../../data/cityscapes/leftImg8bit/val')
    # dec_dataset = PascalContext('../../data/VOCdevkit/VOC2010/JPEGImages')
    # dec_dataset = Ade('../../data/ade/ADEChallengeData2016/images/validation')

    dec_loader = torch.utils.data.DataLoader(
        dec_dataset,
        batch_size=1,
        num_workers=2,
        pin_memory=True
    )

    for i, batch in tqdm(enumerate(dec_loader), total=dec_dataset.__len__()):
        # batch = Image.open('zoneplate.png')
        # batch = [torch.from_numpy(np.array(batch)).unsqueeze(0), '/data']
        if nsct.gpu:
            img = batch[0].unsqueeze(0).float().cuda()
            img_path = batch[1][0]
        else:
            img = batch[0].squeeze(0).numpy()
            img_path = batch[1][0]

        # start_time = time.time()
        y = nsct.dec_iter(img)
        # end_time = time.time()
        # print("Time:%.2fs" % (end_time - start_time))

        # shownsct(y, nsct.gpu)
        save_nsct(y, img_path, nsct.gpu)
        # break
        # a = load_nsct(img_path)
        # b = norm_features(y)
        # b = torch_lst2np_lst(b)
        # a = b



if __name__ == '__main__':
    main()
