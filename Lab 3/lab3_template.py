import scipy.io
#from utils import *
import scipy as scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib import gridspec
from numpy.fft import fftshift, ifftshift, ifft2, fft2
from skimage.filters import window

if __name__ == "__main__":
    # load matlab file
    kdata = scipy.io.loadmat('kdata_phase_error_severe.mat')['kdata']

    # 1.....
    kx, ky = kdata.shape
    PF = 9/16
    N_y = int(ky / PF)
    print(kx, ky, N_y)

    kdata_zpad = np.pad(kdata, ((0,0), (0, N_y - ky)))
    print(kdata_zpad.shape)
    plt.imshow(abs(kdata_zpad), cmap = 'gray', norm = clr.PowerNorm(gamma = 0.2))
    plt.show()

    kdata_herm = np.zeros_like(kdata_zpad)
    kdata_conj = np.rot90(np.conjugate(kdata[:, :N_y //2]), 2)
    kdata_herm[:, :ky] = kdata[:, :ky]
    kdata_herm[:, ky:] = kdata_conj[:, ky-N_y//2:]
    plt.imshow(abs(kdata_herm), cmap = 'gray', norm = clr.PowerNorm(gamma = 0.2))
    plt.show()


    def ifft2c(x, axes=(-2, -1)):
        return np.sqrt(np.size(x)) * fftshift(ifft2(ifftshift(x, axes=axes), axes=axes), axes=axes)


    def fft2c(x, axes=(-2, -1)):
        return np.sqrt(np.size(x)) * fftshift(fft2(fftshift(x, axes=axes), axes=axes), axes=axes)

    recon_zpad = ifft2c(kdata_zpad)
    recon_herm = ifft2c(kdata_herm)
    f, a = plt.subplots(1, 2)
    for i, img in enumerate([recon_zpad, recon_herm]):
        a[i].imshow(abs(img), cmap='gray')
    plt.show()

    # 2...Phase estimation

    def estim_phs(kspace, N):
        '''
        Phase estimation

        Param:
        kspace: asymmetric k-space data
        N: target size of the reconstructed PF dimension
        Return:
        estimated_phase: estimated phase of the input "kspace"
        '''

        kx, ky = kspace.shape
        ref_size = (ky - (N//2)) * 2

        # zero frame with size of (kx, N)
        # Hamming window of (kx,ref_size)
        # plug in phase_ref * hamming to middle of zero frame
        # apply ifft2c and take angle
        # return result

        zero_frame = np.zeros((kx, N)).astype(complex)
        phase_ref = kspace[:, ky - ref_size: ky]
        hamming_window = window("hamm", [kx, ref_size])

        plt.imshow(abs(phase_ref), cmap='gray', norm=clr.PowerNorm(gamma=0.2))
        plt.show()
        plt.imshow(hamming_window)
        plt.show()

        masked = phase_ref * hamming_window
        zero_frame[:, ky - ref_size: ky] = masked
        recon = ifft2c(zero_frame)
        plt.imshow(abs(recon), cmap='gray')

        phase_ref = np.angle(recon)

        return phase_ref

    result_phs_est = estim_phs(kdata, N_y)
    plt.imshow(result_phs_est, cmap='gray', vmin=-np.pi, vmax=np.pi)
    plt.show()

    #3 ... Margosian method:
    def pf_margosian(kdata, N, ftype):
        '''
        Margosian reconstruction for partial Fourier (PF) MRI
        Param:
        kdata: asymmetric k-space data
        N: target size of the reconstructed PF dimension
        ftype: k-space filter ('ramp' or 'hamming')
        Return:
        I: reconstructed magnitude image
        '''

        w = get_ramp(kdata, N)
        kdata_zpad = np.pad(kdata, ((0,0), (0, N_y - ky)))
        i_0 = ifft2c(kdata_zpad * w)

        phs_est = estim_phs(kdata, N)
        exp_ph = np.exp(-1j * phs_est)
        Image = np.abs(exp_ph * i_0)

        return Image




    # 4.. POCS
    def pf_pocs(kdata, N, Nite):
        '''
        POCS reconstruction for partial Fourier (PF) MRI
        Param:
        kdata: asymmetric k-space data
        N: size of the reconstructed PF dimension
        Nite: number of iterations
        Return:
        I: reconstructed magnitude image
        '''

        kx, ky = kdata.shape
        phs_est = estim_phs(kdata, N)
        exp_ph = np.exp()
        kdata_zpad = np.pad(kdata, ((0, 0), (0, N_y - ky)))

        for i in range(Nite):
            '''
               1. ifft to kspace_zpad
               2. Get I_n1
               3. get S_n1 by applying FFT
               4. update kspace_zpad using S_n1
            '''







