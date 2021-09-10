import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import skimage

def get_img_lst(img_root):
    if 'cityscapes' in img_root:
        img_lst = []
        city_dirs = os.listdir(img_root)
        for city in city_dirs:
            imgs_in_city = os.listdir(os.path.join(img_root, city))
            imgs_in_city = [os.path.join(city, img_in_city) for img_in_city in imgs_in_city]
            img_lst.extend(imgs_in_city)
        return img_lst
    if 'VOC' in img_root:
        img_lst = os.listdir(img_root)
        return img_lst
    else:
        raise ValueError('Unsupported dataset type')

def main():
    """
    step1. change img_root
    step2. change noise_mode
    step3. change folder to be saved
    :return:
    """

    noise_mode = 's&p'
    # noise_mode = 'gaussian'

    img_root = '../data/cityscapes/leftImg8bit/val'
    # img_root = '../data/VOCdevkit/VOC2010/JPEGImages'
    img_lst = get_img_lst(img_root)

    for img_name in tqdm(img_lst, total=len(img_lst)):
        img_dir = os.path.join(img_root, img_name)
        img = Image.open(img_dir)
        img = np.asarray(img)
        noisy_img = (skimage.util.random_noise(img, mode=noise_mode, clip=True) * 255).astype(np.uint8)
        img = Image.fromarray(noisy_img)

        folder = os.path.dirname(img_dir).replace('/leftImg8bit', '/leftImg8bit_s&p')
        os.makedirs(folder, exist_ok=True)
        save_dir = os.path.join(folder, os.path.basename(img_dir))
        img.save(save_dir)

if __name__ == '__main__':
    main()
