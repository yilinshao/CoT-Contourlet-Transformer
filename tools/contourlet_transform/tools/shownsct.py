import matplotlib.pyplot as plt
import os
import numpy as np

def shownsct(y, gpu):
    y = norm_features(y)
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


def torch_lst2np_lst(filter_list):
    if isinstance(filter_list, list):
        new_list = []
        for filter_instance in filter_list:
            new_list.append(torch_lst2np_lst(filter_instance))
        return new_list
    else:
        return filter_list.cpu().numpy()


def norm_features(y):
    new_list = []
    if isinstance(y, list):
        for y_i in y:
            new_list.append(norm_features(y_i))
        return new_list
    else:
        y_mean, y_std = y.mean(), y.std()
        y = (y - y_mean) / y_std
        return y


def save_nsct(y, img_dir, gpu):
    folder = os.path.dirname(img_dir).replace('/leftImg8bit', '/nsct')
    os.makedirs(folder, exist_ok=True)

    save_dir = os.path.join(folder, os.path.basename(img_dir).replace('.png', ''))
    y = norm_features(y)
    if gpu:
        y = torch_lst2np_lst(y)
    np.save(save_dir, y)

    # clevels = len(y)
    # for i in range(clevels):
    #     if isinstance(y[i], list):
    #         csubband = len(y[i])
    #         for j in range(csubband):
    #             y[i][j] = ((y[i][j] - np.mean(y[i][j])) / np.std(y[i][j]))
    #             np.save(os.path.join(save_dir, img_name + '{}-{}.png'.format(i, j)), y[i][j])
    #
    #     else:
    #         y[i] = ((y[i] - np.mean(y[i])) / np.std(y[i]))
    #         np.save(os.path.join(save_dir, img_name + '{}.png'.format(i)), y[i])


def load_nsct(img_dir):
    folder = os.path.dirname(img_dir).replace('/leftImg8bit', '/nsct')
    save_dir = os.path.join(folder, os.path.basename(img_dir).replace('.png', '.npy'))
    a = np.load(save_dir, allow_pickle=True)
    a = a.tolist()
    # shownsct(a, False)

    return a


