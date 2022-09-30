from torch.utils import data
import torch
import cv2
from contourlet_dec import ContourletDec
import os
import numpy as np
from PIL import Image
from tools.shownsct import shownsct, save_nsct
from tqdm import tqdm


class CityscapesImages(data.Dataset):
    def __init__(self, root, list_path):
        self.root = root
        self.list_path = list_path

        self.img_list = [line.strip().split()[0] for line in open(os.path.join(root, list_path))]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.img_list[index])).convert('L')
        img = img.resize((512, 256))
        img = np.array(img)
        return img.copy()


def get_index():
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
    return index_dict

def main():
    # get filters
    levels = [0, 1, 2]
    pfilter = 'maxflat'
    dfilter = 'dmaxflat7'
    nsct = ContourletDec(levels, dfilter, pfilter)
    h1, h2, filters = nsct.get_filters()

    # get index
    index_dict = get_index()

    # get dataloader
    dec_dataset = CityscapesImages('../../data/cityscapes', 'train.lst')
    dec_loader = torch.utils.data.DataLoader(
        dec_dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=True
    )

    for i, batch in tqdm(enumerate(dec_loader), total=dec_dataset.__len__()):
        img = batch
        img = img.squeeze(0).numpy()

        clevels = len(levels)
        nIndex = clevels + 1
        y = nsct.dec_iter(img, h1, h2, filters, index_dict, clevels, nIndex)

        save_dir = '/data/nsct_image/'
        os.makedirs(save_dir, exist_ok=True)
        # shownsct(y)
        save_nsct(y, save_dir, img_name=str(i))


if __name__ == '__main__':
    main()

