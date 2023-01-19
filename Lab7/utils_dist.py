'''
Author: Jinho Kim
Email: jinho.kim@fau.de
First created: Nov. 2021
Last modified: 7.Dec.2022
'''

import pathlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from numpy.fft import fftshift, ifftshift, ifft2, fft2
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


def plot_spocks(traj, nSpock):
    f, a = plt.subplots(1, 1)
    a.plot(traj[:, :nSpock].real, traj[:, :nSpock].imag)
    a.set_title(f"k-space trajectory: {nSpock} spokes")

    plt.show()
    plt.close()


def normalize_img(img):
    norm_img = (img - img.mean()) / (img.std())
    return norm_img


def creat_gif(recons: list, title: str, duration: int = 100):
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
        recon = recon / recon.max() * 255  # Rescaling between 0-255
        tmp_img = Image.fromarray(recon).convert("L")
        tmp_img = tmp_img.resize((i * 2 for i in recon.shape))  # double the size
        font = ImageFont.truetype("arial.ttf", size=30)  # Font style
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
