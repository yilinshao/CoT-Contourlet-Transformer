import matplotlib.pyplot as plt
import os
import numpy as np

def shownsct(y):
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


def save_nsct(y, save_dir, img_name):
    clevels = len(y)
    for i in range(clevels):
        if isinstance(y[i], list):
            csubband = len(y[i])
            for j in range(csubband):
                y[i][j] = ((y[i][j] - np.mean(y[i][j])) / np.std(y[i][j]))
                np.save(os.path.join(save_dir, img_name + '{}-{}.png'.format(i, j)), y[i][j])

        else:
            y[i] = ((y[i] - np.mean(y[i])) / np.std(y[i]))
            np.save(os.path.join(save_dir, img_name + '{}.png'.format(i)), y[i])

