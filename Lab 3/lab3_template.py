import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from numpy.fft import fftshift, ifftshift, ifft2, fft2
from skimage.filters import window
import utils_dist as helpers
import scipy
from skimage.metrics import structural_similarity
from skimage.util import compare_images
import math

if __name__ == "__main__":
    # load matlab file
    kdata = scipy.io.loadmat('kdata_phase_error_severe.mat')['kdata']
    kdata_original = scipy.io.loadmat('kdata1.mat')['kdata1']  # k space already

    # 1.....
    kx, ky = kdata.shape
    PF = 9 / 16
    N_y = int(ky / PF)
    print(kx, ky, N_y)

    kdata_zpad = np.pad(kdata, ((0, 0), (0, N_y - ky)))
    print(kdata_zpad.shape)


    kdata_herm = np.zeros_like(kdata_zpad)
    kdata_conj = np.rot90(np.conjugate(kdata[:, :N_y // 2]), 2)
    kdata_herm[:, :ky] = kdata[:, :ky]
    kdata_herm[:, ky:] = kdata_conj[:, ky - N_y // 2:]

    helpers.imshow1row([abs(kdata_zpad),abs(kdata_herm)], ["zero-filled", "Hermitian"], isMag=True, norm=0.2)


    def ifft2c(x, axes=(-2, -1)):
        return np.sqrt(np.size(x)) * fftshift(ifft2(ifftshift(x, axes=axes), axes=axes), axes=axes)


    def fft2c(x, axes=(-2, -1)):
        return np.sqrt(np.size(x)) * fftshift(fft2(fftshift(x, axes=axes), axes=axes), axes=axes)


    recon_zpad = ifft2c(kdata_zpad)
    recon_herm = ifft2c(kdata_herm)
    recon_original = ifft2c(kdata_original)
    helpers.imshow1row([abs(recon_zpad), abs( recon_herm)], ["zero-filled", "Hermitian"], isMag=False)


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
        ref_size = (ky - (N // 2)) * 2

        # zero frame with size of (kx, N)
        # Hamming window of (kx,ref_size)
        # plug in phase_ref * hamming to middle of zero frame
        # apply ifft2c and take angle
        # return result

        zero_frame = np.zeros((kx, N)).astype(complex)
        phase_frame = np.zeros((kx, N)).astype(complex)
        phase_ref = kspace[:, ky - ref_size: ky]
        phase_frame[:, ky - ref_size: ky] = phase_ref
        hamming_window = window("hamm", [kx, ref_size])



        plt.imshow(abs(phase_frame), cmap='gray', norm=clr.PowerNorm(gamma=0.2))
        plt.axis("off")
        plt.title("k-space")
        plt.show()
        plt.imshow(hamming_window)
        plt.axis("off")
        plt.title("window")
        plt.show()

        masked = phase_ref * hamming_window
        zero_frame[:, ky - ref_size: ky] = masked
        recon = ifft2c(zero_frame)
        plt.imshow(abs(recon), cmap='gray')

        phase_ref = np.angle(recon)

        return phase_ref


    result_phs_est = estim_phs(kdata, N_y)
    helpers.imshow1row([result_phs_est], ["Phase estimation"], isMag=False)


    # 3 ... Margosian method:
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

        w = get_ramp(kdata, N) if ftype == 'ramp' else get_hamming(kdata, N)
        kdata_zpad = np.pad(kdata, ((0, 0), (0, N_y - ky)))
        norm = abs(kdata_zpad.max() - kdata_zpad.min())
        i_0 = ifft2c(kdata_zpad * w )


        phs_est = estim_phs(kdata, N)
        exp_ph = np.exp(-1j * phs_est)
        Image = np.abs(exp_ph * i_0) * 1.9

        return Image


    def get_ramp(kdata, N):
        kx, ky = kdata.shape
        sym_gap = ky - (N // 2)
        ramp_start, ramp_end = (N // 2 - sym_gap), (N // 2 + sym_gap)
        ramp = np.zeros((N, N), dtype=kdata.dtype)
        ramp[:, :ramp_start] = np.ones((N, ramp_start))
        ramp_filter = np.tile(np.linspace(1, 0, 2 * sym_gap), (N, 1))
        ramp[:, ramp_start:ramp_end] = ramp_filter

        return ramp


    def get_hamming(kdata, N):
        kx, ky = kdata.shape
        sym_gap = ky - (N // 2)
        hamm_start, hamm_end = (N // 2 - sym_gap), (N // 2 + sym_gap)
        hamm = np.zeros((N, N), dtype=kdata.dtype)
        hamm[:, :hamm_start] = np.ones((N, hamm_start))
        hamm_window = window("hamm", [4 * sym_gap])
        hamm_window = hamm_window[2 * sym_gap:]
        hamm_filter = np.tile(hamm_window, (N, 1))
        hamm[:, hamm_start:hamm_end] = hamm_filter

        return hamm


    margosian_hamm, margosian_ramp = pf_margosian(kdata, N_y, 'hamming'), pf_margosian(kdata, N_y, 'ramp')
    helpers.imshow1row([abs(recon_original), margosian_hamm, margosian_ramp],
                       ["Original", " Margosian(Hamming)", "Margosian(ramp)"], isMag=False)
    helpers.imshow1row([get_ramp(kdata, N_y), get_hamming(kdata, N_y)], ["Ramp filter", "Hamming filter"], isMag=True)
    helpers.imshow1row([helpers.paddedzoom(abs(recon_original)), helpers.paddedzoom(margosian_hamm),
                        helpers.paddedzoom(margosian_ramp)], ["", "", ""], isMag=False)
    helpers.imshow1row([abs(fft2c(margosian_hamm))], isMag=True, norm=0.2)


    # helpers.imshow1row([scipy.ndimage.zoom(result_margosian,  0.0003)], ["Image estimation zoomed"], isMag=False)

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
        exp_ph = np.exp(-1j * phs_est)
        kdata_zpad = np.pad(kdata, ((0, 0), (0, N - ky)))
        kspace = kdata_zpad.copy()
        In = []
        en = []
        Sn = []
        titles = []

        for i in range(Nite):
            '''
               1. ifft to kspace_zpad
               2. Get I_n1
               3. get S_n1 by applying FFT
               4. update kspace_zpad using S_n1
            '''
            I_n0 = ifft2c(kspace)
            I_n1 = exp_ph * abs(I_n0)
            S_n1 = fft2c(I_n1)
            norm = abs(kspace.max() - kspace.min())/(S_n1.max() - S_n1.min())
            print(norm)
            kspace[:,ky:] = S_n1[:,ky:] * norm # 4. update step
            plt.imshow(abs(S_n1), cmap='gray', norm=clr.PowerNorm(gamma=0.2))

            In.append(np.abs(ifft2c(kspace)))
            Sn.append(kspace)
            titles.append("POCS iteration" + str(i+1))

        I = ifft2c(kspace)
        for j in range(len(In)-1):
            if j > 0:
                e_i = compare_images(In[j+1], In[j], method='diff')
                en.append(e_i)
            else:
                e_i = compare_images(In[j], np.abs(ifft2c(kdata_zpad)), method='diff')
                en.append(e_i)


        return np.abs(I), Sn, In, en, titles


    result_POCS, k_evolve, I_evolve, error, titles = pf_pocs(kdata, N_y, 10)
    print(titles)
    helpers.imshow1row([result_POCS], ["Image estimation POCS"], isMag=False)
    helpers.imshow1row(k_evolve[::2], titles[1::2], isMag=True, norm=0.2)
    helpers.imshow1row(I_evolve[::2], titles[1::2], isMag=False)
    helpers.imshow1row(error[::2], titles[1::2], isMag=False)
    helpers.imshow1row([helpers.paddedzoom(I_evolve[1]),helpers.paddedzoom(I_evolve[3]),helpers.paddedzoom(I_evolve[5]),
                        helpers.paddedzoom(I_evolve[7]), helpers.paddedzoom(I_evolve[9])]
                       , titles[1::2], isMag=False)


    def calc_perf(original, titles: list, results: list, rmse: bool = True, ssim: bool = True, snr: bool = True, psnr: bool = True):

        perf = np.zeros([4,len(results)])
        for i, img in enumerate(results):
            if rmse:
                MSE = np.square(np.subtract(original, img)).mean()
                RMSE = math.sqrt(MSE) * 1000
            if ssim:
                SSIM = structural_similarity(original, img, multichannel=False)
            if snr:
                SNR = helpers.snr_calc(img, 100)
            if psnr:
                if MSE == 0:
                    PSNR = 0
                else:
                    PSNR = 10 * math.log10(1. / MSE)

            perf[:, i] = np.array((RMSE, SSIM, SNR, PSNR))
        res = np.vstack((np.array(titles), perf))
        print(res)
        return res

    perf_res_POCS = calc_perf(abs(recon_original), titles[1::2],I_evolve[::2])
    perf_res = calc_perf(abs(recon_original),['Original', 'Zero-filled','Hermitian', 'Margosian(ramp)','Margosian(Hamming)', 'POCS 9'],
                                [np.abs(recon_original),np.abs( recon_zpad),np.abs(recon_herm),margosian_ramp,margosian_hamm ,I_evolve[8]])
    helpers.imshow1row([abs(recon_original),abs( recon_zpad),np.abs(recon_herm),margosian_ramp,margosian_hamm ,I_evolve[8]],
                       ['Original', 'Zero-filled','Hermesian', 'Margosian(ramp)','Margosian(Hamming)', 'POCS 9'],
                       isMag=False)
    helpers.imshow1row([helpers.paddedzoom(abs(recon_original)), helpers.paddedzoom(abs(recon_zpad)),
                        helpers.paddedzoom(np.abs(recon_herm)),  helpers.paddedzoom(margosian_ramp),
                        helpers.paddedzoom(margosian_hamm),helpers.paddedzoom(I_evolve[8])],
                       ['Original', 'Zero-filled','Hermitian', 'Margosian(ramp)','Margosian(Hamming)', 'POCS 9'], isMag=False)

    diff_images = helpers.diff_images(abs(recon_original), [abs( recon_zpad),np.abs(recon_herm),margosian_ramp,margosian_hamm ,I_evolve[8]])
    helpers.imshow1row(diff_images, isMag=False)



















