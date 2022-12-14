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
        figure.set_size_inches(15, 15)
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

    # todo: return mc
    # return mc


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

    # todo: return ir and g
    # return ir, g


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
