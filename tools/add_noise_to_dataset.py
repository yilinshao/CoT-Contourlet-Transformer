from torch.utils import data
import torch
import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import skimage


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

def get_pascal_context_lst(root):
    img_list = os.listdir(root)
    return os.listdir(root)


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

    # get dataloader
    # dec_dataset = CityscapesImages('../../data/cityscapes/leftImg8bit/test')
    # dec_dataset = PascalContext('../../data/VOCdevkit/VOC2010/JPEGImages')
    # dec_dataset = Ade('../../data/ade/ADEChallengeData2016/images/validation')

    root = '../data/VOCdevkit/VOC2010/JPEGImages'
    noise_mode = 's&p'
    # noise_mode = 'gaussian'
    img_lst = os.listdir(root)
    for img_name in tqdm(img_lst, total=len(img_lst)):
        img_dir = os.path.join(root, img_name)
        img = Image.open(img_dir)
        img = np.asarray(img)
        noisy_img = (skimage.util.random_noise(img, mode=noise_mode, clip=True) * 255).astype(np.uint8)
        img = Image.fromarray(noisy_img)

        folder = os.path.dirname(img_dir).replace('/JPEGImages', '/JPEGImages_s&p')
        os.makedirs(folder, exist_ok=True)
        save_dir = os.path.join(folder, os.path.basename(img_dir))
        img.save(save_dir)

if __name__ == '__main__':
    main()
