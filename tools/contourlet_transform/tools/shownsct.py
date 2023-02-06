import matplotlib.pyplot as plt
import os
import numpy as np



def shownsct(y, gpu):
    # y = norm_features(y)
    if gpu:
        y = torch_lst2np_lst(y)
    clevels = len(y)
    for i in range(clevels):
        if isinstance(y[i], list):
            csubband = len(y[i])
            if csubband > 7:
                col = 4
            else:
                col = 2
            row = int(csubband/col)
            for j in range(csubband):
                plt.rcParams['figure.dpi'] = 300
                plt.subplot(row, col, j + 1)
                plt.imshow(y[i][j], cmap='gray')
                plt.title('NSSC coefficients: level {}'.format(i))
            plt.show()
        else:
            plt.rcParams['figure.dpi'] = 300
            plt.imshow(y[i], cmap='gray')
            plt.title('Nonsubsampled Contourlet coefficients level {}'.format(i))
            plt.show()

def gray_to_rgb(gray_img, th):
    gray_img = (gray_img - np.min(gray_img))/(np.max(gray_img) - np.min(gray_img))
    rgb_shape = gray_img.shape + (3,)
    rgb_img = np.zeros(rgb_shape, dtype=np.int)
    rgb_img[np.where(gray_img < th)] = [255, 255, 255]
    rgb_img[np.where(gray_img >= th)] = [0, 0, 0]

    return rgb_img

def save_rgb(y, gpu, rgb_dir):
    threshold = [0.5, 0.48, [0.52, 0.53], [0.55, 0.5, 0.53, 0.52]]
    os.makedirs(rgb_dir, exist_ok=True)
    # y = norm_features(y)
    if gpu:
        y = torch_lst2np_lst(y)
    clevels = len(y)
    for i in range(clevels):
        if isinstance(y[i], list):
            for j in range(len(y[i])):
                save_name = os.path.join(rgb_dir, 'lp_{}_dfb_{}_th_{}.jpg'.format(i, j, threshold[i][j]))
                save_img = gray_to_rgb(y[i][j], threshold[i][j])
                plt.imshow(save_img)
                plt.axis('off')
                plt.savefig(save_name, dpi=600)
                # plt.show()
        elif i == 0:
            save_name = os.path.join(rgb_dir, 'low_freq_component.jpg'.format(i, threshold[i]))
            plt.imshow(y[i], cmap='gray')
            plt.axis('off')
            plt.savefig(save_name, dpi=600)
        else:
            save_name = os.path.join(rgb_dir, 'lp_{}_dfb_0_th_{}.jpg'.format(i, threshold[i]))
            save_img = gray_to_rgb(y[i], threshold[i])
            plt.imshow(save_img)
            plt.axis('off')
            plt.savefig(save_name, dpi=600)
            # plt.show()


def torch_lst2np_lst(filter_list):
    if isinstance(filter_list, list):
        new_list = []
        for filter_instance in filter_list:
            new_list.append(torch_lst2np_lst(filter_instance))
        return new_list
    elif filter_list.ndim == 4:
        return filter_list[0, 0].cpu().numpy()
    else:
        return filter_list.cpu().numpy()


def np_lst2np_stack(feature_lst):
    out = np.expand_dims(feature_lst[0], axis=2)
    for feature in feature_lst[1:]:
        if isinstance(feature, list):
            for direction in feature:
                out = np.concatenate((out, np.expand_dims(direction, axis=2)), axis=2)
        else:
            out = np.concatenate((out, np.expand_dims(feature, axis=2)), axis=2)
    return out


def norm_features(y):
    new_list = []
    if isinstance(y, list):
        for y_i in y:
            new_list.append(norm_features(y_i))
        return new_list
    else:
        y_mean, y_std = y.mean(), y.std()
        y = (y - y_mean) / y_std

        # y_max, y_min = y.max(), y.min()
        # y = (y - y_min)/(y_max - y_min)
        #
        # if y.max() != 1:
        #     print(y)

        return y

def save_nsct(y, img_dir, gpu):
    # print(img_dir)
    folder = os.path.dirname(img_dir).replace('/images', '/nsct_222')
    rgb_folder = os.path.dirname(img_dir).replace('/slk_JPEGImages', '/nsct_rgb')
    # folder = os.path.dirname(img_dir).replace('/JPEGImages', '/nsct_01')
    # folder = os.path.dirname(img_dir).replace('/leftImg8bit', '/nsct_01')

    os.makedirs(folder, exist_ok=True)
    os.makedirs(rgb_folder, exist_ok=True)

    save_dir = os.path.join(folder, os.path.basename(img_dir).replace('.png', '').replace('.jpg', ''))
    save_rgb_dir = os.path.join(rgb_folder, os.path.basename(img_dir).replace('.png', '').replace('.jpg', ''))

    # =================================save RGB coefficients======================================= #
    save_rgb(y, gpu, save_rgb_dir)
    # ============================================================================================= #

    y = norm_features(y)
    if gpu:
        y = torch_lst2np_lst(y)
        y = np_lst2np_stack(y)

    # np.save(save_dir, y)






def load_nsct(img_dir):
    folder = os.path.dirname(img_dir).replace('/leftImg8bit', '/nsct')
    save_dir = os.path.join(folder, os.path.basename(img_dir).replace('.png', '.npy'))
    a = np.load(save_dir, allow_pickle=True)
    a = a.tolist()
    # shownsct(a, False)

    return a


