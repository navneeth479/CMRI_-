'''
Author: Jinho Kim
E-mail: jiinho.kim@fau.de
'''

from numpy.fft import fftshift, ifftshift, ifft2, fft2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cv2
from skimage.util import compare_images


def imshow1row(imgs: list, titles: list = None, isMag: bool = True, filename: str = None, log: bool = False,
               norm: float = 1.0):
    """
        Plot images in one row
        @param imgs: images in a list (e.g., [img1,(img2, img3,...)]
        @param titles: titles in a list (optional)
        @param isMag: plot images in magnitude scale or not (optional, default=True)
        @param filename: if given, save the plot to filename, otherwise, plot in an window (optional)
        @param log: plot images in a logarithm scale (optional, default=False)
        @param norm: Adjust image intensity levels. (Recommend: for k-space, norm=0.2~0.3)
    """
    f, a = plt.subplots(1, len(imgs), figsize=(8, 8))
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


def paddedzoom(img, zoomfactor=3.5):
    '''
    Zoom in/out an image while keeping the input image shape.
    i.e., zero pad when factor<1, clip out when factor>1.
    there is another version below (paddedzoom2)
    '''

    out = np.zeros_like(img)
    zoomed = cv2.resize(img, None, fx=zoomfactor, fy=zoomfactor)

    h, w = img.shape
    zh, zw = zoomed.shape

    if zoomfactor < 1:  # zero padded
        out[(h - zh) // 2:-(h - zh) // 2, (w - zw) // 2:-(w - zw) // 2] = zoomed
    else:  # clip out
        out = zoomed[180 + (zh - h) // 2: 180 - (zh - h) // 2, 120 + (zw - w) // 2: 120 - (zw - w) // 2]

    return out


def crop(img, nRO, os_rate):
    x_os, y_os = img.shape
    nRO = nRO // os_rate
    x, y = int((x_os - nRO) // 2), int((x_os + nRO) // 2)
    img = img[x:y, x:y]

    return img


def snr_calc(image, size):
    frame1 = np.zeros_like(image)
    frame2 = np.zeros_like(image)
    frame3 = np.zeros_like(image)
    frame4 = np.zeros_like(image)
    frame5 = np.zeros_like(image)
    x_axis = y_axis = np.arange(0, image.shape[0], 1)
    x_axisv, y_axisv = np.meshgrid(x_axis, y_axis)

    mask1 = np.ones([size, size])
    mask2 = ((x_axisv - 256) ** 2 + (y_axisv - 256) ** 2 < size ** 2)
    shift = 200
    frame_size = frame1.shape[0]

    x, y = (frame_size - size) // 2, (frame_size + size) // 2
    frame1[x + shift:y + shift, x + shift:y + shift] = mask1
    frame2[x + shift:y + shift, x - shift:y - shift] = mask1
    frame3[x - shift:y - shift, x + shift:y + shift] = mask1
    frame4[x - shift:y - shift, x - shift:y - shift] = mask1
    frame5 = mask2

    n1 = image * frame1
    n2 = image * frame2
    n3 = image * frame3
    n4 = image * frame4
    s1 = image * frame5
    noise = np.array((n1, n2, n3, n4))
    # imshow1row([image, n1, n2, n3, n4, s1],isMag=False)
    std = []

    for i in range(len(noise)):
        std.append(np.std(noise[i]))

    SNR = np.average(s1) / np.average(std)
    return SNR


def fft2c(x, axes=(-2, -1)):
    """
    FFT for MRI data
    @param x: MRI image to be Fourier transformed (shape: [..., x, y])
    @param axes: Axes of row and col
    @return: FFT(image)
    """
    return (1 / np.sqrt(np.size(x))) * fftshift(fft2(ifftshift(x, axes=axes), axes=axes), axes=axes)


def ifft2c(x, axes=(-2, -1)):
    """
    IFFT for kdata
    @param x: kdata to be inverse Fourier transformed (shape: [..., x, y])
    @param axes: Axes of read-out and phase encoding direction
    @return: IFFT(kdata)
    """
    return np.sqrt(np.size(x)) * fftshift(ifft2(ifftshift(x, axes=axes), axes=axes), axes=axes)


def diff_images(original, img: list):
    e = []
    for j in range(len(img)):
        e_i = compare_images(img[j], original, method='diff')
        e.append(e_i)
    return e
