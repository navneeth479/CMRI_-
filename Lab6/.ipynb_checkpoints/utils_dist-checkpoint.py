'''
Author: Jinho Kim
Email: jinho.kim@fau.de
First created: Nov. 2021
Last modified: 04. Dec. 2022
'''

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftshift, ifftshift, ifft2, fft2
from scipy.linalg import fractional_matrix_power


def imshow1row(imgs: list, titles: list = None, isMag: bool = True, filename: str = None, log: bool = False,
               norm: float = 1.0):
    """
        Plot images in one row
        @param imgs: images in a list
        @param titles: titles in a list (optional)
        @param isMag: plot images in magnitude scale or not (optional, default=True)
        @param filename: if given, save the plot to filename, otherwise, plot in an window (optional)
        @param log: plot images in a logarithm scale (optional, default=False)
        @param norm: Adjust image intensity levels. (Recommend: for k-space, norm=0.2~0.3)
    """
    f, a = plt.subplots(1, len(imgs))

    titles = [None] * len(imgs) if titles is None else titles

    for i, (img, title) in enumerate(zip(imgs, titles)):
        ax = a[i] if len(imgs) >= 2 else a
        img = abs(img) if isMag else img
        img = np.log2(img + 1) if log else img
        ax.imshow(img, cmap='gray', norm=colors.PowerNorm(gamma=norm))
        ax.axis('off')
        ax.set_title(title)

    if filename is None:
        plt.show()
    elif filename is not None:
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(28, 14)
        plt.savefig(filename, bbox_inches='tight')
    plt.close(f)


def fft2c(x, axes=(-2, -1)):
    return (1 / np.sqrt(np.size(x))) * fftshift(fft2(ifftshift(x, axes=axes), axes=axes), axes=axes)


def ifft2c(x, axes=(-2, -1)):
    return np.sqrt(np.size(x)) * fftshift(ifft2(ifftshift(x, axes=axes), axes=axes), axes=axes)
