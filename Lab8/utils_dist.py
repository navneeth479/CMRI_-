'''
A package containing utility functions for the computational MRI exercise.

Author          : Jinho Kim
Email           : jinho.kim@fau.de
First created   : Dec. 2021
Last update     : Mon. Dec. 19 2022
'''
import math
import pathlib

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pywt
from PIL import Image, ImageDraw, ImageFont
from numpy.fft import fftshift, ifftshift, ifft2, fft2
from tqdm.auto import tqdm
import cv2
from skimage.util import compare_images


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
        # ax.text(0.8, 0.95, 'Begin text', horizontalalignment='center', verticalalignment='center', color='yellow',
        # transform=ax.transAxes)
        ax.imshow(img, cmap='gray', norm=colors.PowerNorm(gamma=norm))
        ax.axis('off')
        ax.set_title(title)

    if filename is None:
        plt.show()
    elif filename is not None:
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(14, 14)
        plt.savefig(filename, bbox_inches='tight')
    plt.close(f)


def imshow_metric(imgs: list, titles: list = None, isMag: bool = True, filename: str = None, log: bool = False,
                  norm: float = 1.0, val: str = None, metrics: list = None):
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
        ax.text(0.8, 0.95, val + '=' + str(metrics[i]), horizontalalignment='center', verticalalignment='center',
                color='yellow',
                transform=ax.transAxes)
        ax.imshow(img, cmap='gray', norm=colors.PowerNorm(gamma=norm))
        ax.axis('off')
        ax.set_title(title)

    if filename is None:
        plt.show()
    elif filename is not None:
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(14, 14)
        plt.savefig(filename, bbox_inches='tight')
    plt.close(f)


def fft2c(x, axes=(-2, -1)):
    return (1 / np.sqrt(np.size(x))) * fftshift(fft2(ifftshift(x, axes=axes), axes=axes), axes=axes)


def ifft2c(x, axes=(-2, -1)):
    return np.sqrt(np.size(x)) * fftshift(ifft2(ifftshift(x, axes=axes), axes=axes), axes=axes)


def dwt2(x, wavelet='db4', mode='periodization', level=4, axes=(-2, -1)):
    """Discrete 2D wavelet transform

    Args:
        x (ndarray): data to be transformed
        wavelet (str, optional): wavelet to use. Defaults to 'db4'.
        mode (str, optional): Signal extension mode. Defaults to 'periodization'.
        level (int, optional): Decomposition level. Defaults to 4.
        axes (tuple, optional): Axis over which to compute the DWT. Defaults to (-2, -1).

    Returns:
        Complex wavelet transformed data
    """
    c_real = pywt.wavedec2(np.real(x), wavelet=wavelet, mode=mode, level=level, axes=axes)
    c_imag = pywt.wavedec2(np.imag(x), wavelet=wavelet, mode=mode, level=level, axes=axes)
    c_real_array, s = pywt.coeffs_to_array(c_real)
    c_imag_array, _ = pywt.coeffs_to_array(c_imag)

    return c_real_array + 1j * c_imag_array, s


def idwt2(x, s, wavelet='db4', mode='periodization'):
    """Inverse discrete 2D wavelet transform

    Args:
        x (ndarray): Approximated coefficients
        s (list): Coefficient details
        wavelet (str, optional): wavelet to use. Defaults to 'db4'.
        mode (str, optional): Signal extension mode. Defaults to 'periodization'.

    Returns:
        Complex inverse wavelet transformed data
    """
    c_real_thresh = pywt.array_to_coeffs(np.real(x), s, output_format='wavedec2')
    c_imag_thresh = pywt.array_to_coeffs(np.imag(x), s, output_format='wavedec2')
    rec_real = pywt.waverec2(c_real_thresh, wavelet=wavelet, mode=mode)
    rec_imag = pywt.waverec2(c_imag_thresh, wavelet=wavelet, mode=mode)

    return rec_real + 1j * rec_imag


def compress(x, comp_factor: int):
    """Compress approximated coefficients by the comp_factor

    Args:
        x (ndarray): Approximated coefficients
        comp_factor (int): Compression factor

    Returns:
        x: Compressed approximated coefficients
    """
    sorted_x = np.flip(np.sort(np.abs(x), None))  # descending order sorted
    idx = len(sorted_x) // comp_factor
    threshold = sorted_x[idx]
    x[abs(x) < threshold] = 0

    return x


def SoftT(x, t: float):
    """Soft-threshold

    Args:
        x (ndarray): Approximated coefficients
        t (float): Regularization parameter

    Returns:
        Soft-thresholded coefficients
    """
    return x / abs(x) * np.maximum((abs(x) - t), 0)


def cs_ista(data, lamda: float, n_it: int):
    """Compressed Sensing using the iterative soft-threshold algorithm

    Args:
        data (ndarray): kspace
        lamda (float): Regularization parameter
        n_it (int): Maximum iteration number

    Returns:
        m (ndarray): CS reconstructed image
        inter_m (list): A list containing the initial solution and the CS reconstructed image
    """

    lamda /= 100
    threshold = lamda * np.max(np.abs(ifft2c(data)))
    inter_m = []
    cost_list = []
    mask = abs(data) > 0
    m = ifft2c(data)  # initial solution

    with tqdm(total=n_it, unit='iter', leave=True) as pbar:
        for it in range(n_it):
            inter_m.append(abs(m))

            # enforce sparsity....
            c, s = dwt2(m)

            # Soft thresholding....
            soft_th = SoftT(c, threshold)

            # Go back to k-space....
            m = idwt2(soft_th, s)
            m_k = fft2c(m)

            # enforcing data consistency....
            dc = m_k * mask - data
            m -= ifft2c(dc)

            cost = np.sqrt(np.sum(np.square(abs(dc)))) + lamda * np.sum(abs(c))
            cost_list.append(cost)
            pbar.set_description(desc=f'Iteration {it: 2d}')
            pbar.set_postfix({"Cost": f"{cost:.5f}"})
            pbar.update()

    inter_m.append(abs(m))

    return m, inter_m, cost_list


def create_gif(recons: list, title: str, duration: int = 100):
    '''
    Create a gif from the list of images.
    The gif image shows double the size of the reconstructed images.
    @param recons: [recon1, recon2, ..., reconN]
        Type        : list
        reconN      : ndarray (shape: rxc) containing float values
    @param title: Title of gif file
    @param duration: duration of gif
    '''
    recon_gif = []
    for i, recon in enumerate(recons):
        recon = recon / recon.max() * 255
        tmp_img = Image.fromarray(recon).convert("L")
        tmp_img = tmp_img.resize((i * 2 for i in recon.shape))

        # For Mac OS
        # font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 30)
        # For Windows
        font = ImageFont.truetype("arial.ttf", 30)

        draw = ImageDraw.Draw(tmp_img)
        draw.text((0, 0), f'iter={i}', fill=255, font=font)
        recon_gif.append(tmp_img)

    title = pathlib.Path(title)
    title = f"{title}.gif" if title.suffix != 'gif' else title
    recon_gif[0].save(f'{title}', format="GIF", append_images=recon_gif[1:], save_all=True, duration=duration,
                      loop=0)


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


def diff_images(original, img: list):
    e = []
    for j in range(len(img)):
        norm_img = abs((img[j] - img[j].mean()) / (img[j].std()))
        e_i = compare_images(abs(norm_img), original, method='diff')
        e.append(e_i)
    return e


def normalize_img(img):
    norm_img = (img - img.mean()) / (img.std())
    return norm_img


def calc_perf(original, results: list, rmse: bool = False, l1: bool = False):
    perf = []
    for i, img in enumerate(results):
        #norm_img = (img - img.mean()) / img.std()
        if rmse:
            MSE = np.square(np.subtract(abs(original), abs(img))).mean()
            RMSE = math.sqrt(MSE)
            perf.append(round(RMSE * 10, 5))
        if l1:
            L1 = np.sum(abs(img))
            perf.append(round(L1, 4))

    print(perf)
    return perf
