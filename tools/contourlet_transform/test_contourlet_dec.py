import numpy

from contourlet_dec import ContourletDec
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from tools.shownsct import shownsct
from PIL import Image


def main(im):
    print('Displaying the input image...')
    plt.rcParams['figure.dpi'] = 300
    plt.imshow(im, cmap='gray')
    plt.title("Input Image")
    plt.show()

    # Parameteters:
    levels = [0, 1, 2]
    pfilter = 'maxflat'
    dfilter = 'dmaxflat7'

    nsct = ContourletDec(levels, dfilter, pfilter)
    coeffs = nsct.nsctdec(im, gpu_mode=False)

    # Display the coefficients
    print('Displaying the contourlet coefficients...')
    shownsct(coeffs)


if __name__ == '__main__':
    # img = np.asarray(io.imread('zoneplate.png'), dtype=np.float64)

    img = Image.open('/document/detr/data/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png').convert('L')
    img = img.resize((256, 128))
    img = numpy.array(img)
    main(img)


