'''
Author: Jinho Kim
Email: jinho.kim@fau.de
Last update:
'''
from numpy.fft import fftshift, ifftshift, ifft2, fft2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.linalg import fractional_matrix_power, pinv
import cv2
import math
from skimage.util import compare_images
from skimage.metrics import structural_similarity


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
        :param rmse:
    """
    f, a = plt.subplots(1, len(imgs))

    titles = [None] * len(imgs) if titles is None else titles

    for i, (img, title) in enumerate(zip(imgs, titles)):
        ax = a[i] if len(imgs) >= 2 else a
        img = abs(img) if isMag else img
        img = np.log2(img + 1) if log else img
        ax.imshow(img, cmap='gray', norm=colors.PowerNorm(gamma=norm))
        ax.axis('off')
        ax.set_title(title, fontsize=100)
        plt.text(230, 230, s='0.0', bbox=dict(fill=True, edgecolor='yellow', linewidth=2))

    if filename is None:
        plt.show()
    elif filename is not None:
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(12, 10)
        plt.savefig(filename, bbox_inches='tight')
    plt.close(f)


def fft2c(x, axes=(-2, -1)):
    return (1 / np.sqrt(np.size(x))) * fftshift(fft2(ifftshift(x, axes=axes), axes=axes), axes=axes)


def ifft2c(x, axes=(-2, -1)):
    return np.sqrt(np.size(x)) * fftshift(ifft2(ifftshift(x, axes=axes), axes=axes), axes=axes)


def sos_comb(m):
    '''
    Sum of square

    :param m: multicoil images [nPE, nFE, nCh]

    :return: mc: combined image [nPE, nFE]
    '''

    mean = np.mean(m, axis=2)
    img = []
    for i in range(m.shape[2]):
        s = np.power((m[:, :, i] - mean), 2)
        img.append(s)
    print(len(img))
    mc = sum(img)
    return mc


def ls_comb(m, c, PSI=None):
    '''
    Least-squares (matched filter)

    :param m: multicoil images [nPE,nFE,nCh]
    :param c: coil sensitivity maps [nPE,nFE,nCh]
    :param PSI: noise correlation matrix [nCh, nCh] (optional [None or PSI])

    :return: mc: combined image [nPE,nFE]
    '''

    if PSI is not None:
        psi_is = fractional_matrix_power(PSI, -1 / 2)
        c = (psi_is @ c.transpose(0, 2, 1)).transpose(0, 2, 1)
        m = (psi_is @ m.transpose(0, 2, 1)).transpose(0, 2, 1)
    numerator = np.sum(c.conj() * m, axis=-1)
    denominator = np.linalg.norm(c, axis=-1)
    mc = np.divide(numerator, denominator, where=(denominator!=0))
    return mc


def calc_g(C):
    '''
    Calculate g-factor
    @param C: C^H*PSI^(-1)*C
    @return: g-factor
    '''
    g = np.sqrt(np.real(np.diag(pinv(np.conj(C.T) @ C)) * np.diag(np.conj(C.T) @ C)))
    return g


def sense_recon(ia, c, PSI, R):
    '''
    SENSE reconstruction

    :param ia: multicoil aliased images [nPE//2,nFE,nCh]
    :param c: coil sensitivity maps [nPE,nFE,nCh]
    :param PSI: noise correlation matrix [nCh, nCh]
    :param R: acceleration factor

    :param ir: unaliased image [nPE,nFE]
    :param g: g-factor map [nPE,nFE]
    '''

    PE, RO, nCoil = c.shape
    psi_is = fractional_matrix_power(PSI, -1/2)
    c = (psi_is @ c.transpose(0,2,1)).transpose(0,2,1)
    ia = (psi_is @ ia.transpose(0,2,1)).transpose(0,2,1)

    dim = PE
    nonzero_idx = np.array(np.nonzero(np.sum(c,axis=-1)))
    ir = np.zeros((PE, RO), dtype=ia.dtype)
    g = np.zeros((PE, RO))

    for idx_PE, idx_RO in zip(*nonzero_idx):
        if not ir[idx_PE, idx_RO]:
            locs = np.mod(np.arange(idx_PE, idx_PE + dim, dim/R, dtype=int), dim)
            C = c[locs, idx_RO, :].T
            C_pinv = pinv(C)
            idx_PE_alias = get_alias_idx(PE, R, locs)
            assert idx_PE_alias is not None
            ir[locs, idx_RO] = C_pinv @ ia[idx_PE_alias, idx_RO, :]
            g[locs, idx_RO] = calc_g(C)

    return ir, g




def get_alias_idx(PE, R, locs):
    '''
    Get an index for aliased image among indices in locs
    @param PE: Length of phase encoding
    @param R: Acceleration factor
    @param locs: indices for SENSE reconstruction at one point
    @return: an index for aliased image
    '''
    alias_PE = np.arange(0, PE, R).size
    min_idx = PE // 2 - alias_PE // 2
    max_idx = PE // 2 + alias_PE // 2
    for loc in locs:
        if min_idx <= loc < max_idx:
            return loc - min_idx
    else:
        return None

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
        out = zoomed[180 + (zh - h) // 2: 180 -(zh - h) // 2, 120 + (zw - w) // 2: 120 -(zw - w) // 2]

    return out

def diff_images(original,  img:list):
    e = []
    for j in range(len(img)):
        e_i = compare_images(img[j], original, method='diff')
        e.append(e_i)
    return e

def calc_perf(original, titles: list, results: list, rmse: bool = True, ssim: bool = True):

    perf = np.zeros([2,len(results)])
    for i, img in enumerate(results):
        if rmse:
            MSE = np.square(np.subtract(abs(original), abs(img))).mean()
            RMSE = math.sqrt(MSE)
        if ssim:
            SSIM = structural_similarity(original, img, multichannel=False)

        perf[:, i] = np.array((RMSE, SSIM))
    res = np.vstack((np.array(titles), perf))
    print(res)
    return res
