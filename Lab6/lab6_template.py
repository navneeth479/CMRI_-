'''
Author: Jinho Kim
Email: jinho.kim@fau.de
First created: Tue Dec 6 22:16 2022 CET
Last modified: Tue Dec 6 22:16 2022 CET
'''
import scipy.io

from grappa_dist import GRAPPA
from utils_dist import *

if __name__ == "__main__":
    mat = scipy.io.loadmat('data_brain_8coils.mat')
    kdata = mat['d']  # (PE,RO,nCoil)
    sens_maps = mat['c']  # (PE,RO,nCoil)
    noise_maps = mat['n']  # (RO,nCoil)
    PSI = np.cov(noise_maps, rowvar=False)  # (nCoil, nCoil)


    # ....Experiment (3.b)
    recon = ifft2c(kdata, axes=(0, 1))
    matched_filter = ls_comb(recon, sens_maps, PSI)
    matched_filter_norm1 = (matched_filter - matched_filter.mean()) / (matched_filter.std())
    # ...........ACS= 24, kernel = vary................
    g23 = GRAPPA(kdata.copy(), nACS=24, kernel_size=(2, 3))
    g45 = GRAPPA(kdata.copy(), nACS=24, kernel_size=(4, 5))
    g67 = GRAPPA(kdata.copy(), nACS=24, kernel_size=(6, 7))

    gg23R4 = g23.grappa(4, flag_acs=True)
    recon_gg23R4 = ls_comb(ifft2c(gg23R4, axes=(0, 1)), sens_maps, PSI)
    gg45R4 = g45.grappa(4, flag_acs=True)
    recon_gg45R4 = ls_comb(ifft2c(gg45R4, axes=(0, 1)), sens_maps, PSI)
    gg67R4 = g67.grappa(4, flag_acs=True)
    recon_gg67R4 = ls_comb(ifft2c(gg67R4, axes=(0, 1)), sens_maps, PSI)
    imshow1row([ recon_gg23R4, recon_gg45R4, recon_gg67R4], [ 'K=2x3| R=4| ACS=24', 'K=4x5| R=4| ACS=24', 'K=6x7| R=4| ACS=24'], filename='3b_24')

    error1 = diff_images(abs(matched_filter_norm1), [recon_gg23R4, recon_gg45R4, recon_gg67R4])
    imshow1row(error1, ['K=2x3| R=4| ACS=24', 'K=4x5| R=4| ACS=24', 'K=6x7| R=4| ACS=24'], filename='e_1')
    perf1 = calc_perf(matched_filter_norm1, ['K=2x3| R=4| ACS=24', 'K=4x5| R=4| ACS=24', 'K=6x7| R=4| ACS=24'], [ recon_gg23R4, recon_gg45R4, recon_gg67R4], ssim=False)

    # ...........ACS= 36, kernel = vary................
    g23 = GRAPPA(kdata.copy(), nACS=36, kernel_size=(2, 3))
    g45 = GRAPPA(kdata.copy(), nACS=36, kernel_size=(4, 5))
    g67 = GRAPPA(kdata.copy(), nACS=36, kernel_size=(6, 7))

    gg23R4 = g23.grappa(4, flag_acs=True)
    recon_gg23R4 = ls_comb(ifft2c(gg23R4, axes=(0, 1)), sens_maps, PSI)
    gg45R4 = g45.grappa(4, flag_acs=True)
    recon_gg45R4 = ls_comb(ifft2c(gg45R4, axes=(0, 1)), sens_maps, PSI)
    gg67R4 = g67.grappa(4, flag_acs=True)
    recon_gg67R4 = ls_comb(ifft2c(gg67R4, axes=(0, 1)), sens_maps, PSI)
    imshow1row([ recon_gg23R4, recon_gg45R4, recon_gg67R4], [ 'K=2x3| R=4| ACS=36', 'K=4x5| R=4| ACS=36', 'K=6x7| R=4| ACS=36'], filename='3b_36')

    error2 = diff_images(abs(matched_filter_norm1), [recon_gg23R4, recon_gg45R4, recon_gg67R4])
    imshow1row(error2, ['K=2x3| R=4| ACS=36', 'K=4x5| R=4| ACS=36', 'K=6x7| R=4| ACS=36'], filename='e_2')
    perf2 = calc_perf(matched_filter_norm1, ['K=2x3| R=4| ACS=36', 'K=4x5| R=4| ACS=36', 'K=6x7| R=4| ACS=36'],
                     [recon_gg23R4, recon_gg45R4, recon_gg67R4], ssim=False)

    # ...........ACS= 48, kernel = vary................
    g23 = GRAPPA(kdata.copy(), nACS=48, kernel_size=(2, 3))
    g45 = GRAPPA(kdata.copy(), nACS=48, kernel_size=(4, 5))
    g67 = GRAPPA(kdata.copy(), nACS=48, kernel_size=(6, 7))

    gg23R4 = g23.grappa(4, flag_acs=True)
    recon_gg23R4 = ls_comb(ifft2c(gg23R4, axes=(0, 1)), sens_maps, PSI)
    gg45R4 = g45.grappa(4, flag_acs=True)
    recon_gg45R4 = ls_comb(ifft2c(gg45R4, axes=(0, 1)), sens_maps, PSI)
    gg67R4 = g67.grappa(4, flag_acs=True)
    recon_gg67R4 = ls_comb(ifft2c(gg67R4, axes=(0, 1)), sens_maps, PSI)
    imshow1row([recon_gg23R4, recon_gg45R4, recon_gg67R4],
               ['K=2x3| R=4| ACS=48', 'K=4x5| R=4| ACS=48', 'K=6x7| R=4| ACS=48'], filename='3b_48')

    error3 = diff_images(abs(matched_filter_norm1), [recon_gg23R4, recon_gg45R4, recon_gg67R4])
    imshow1row(error3, ['K=2x3| R=4| ACS=48', 'K=4x5| R=4| ACS=48', 'K=6x7| R=4| ACS=48'], filename='e_3')
    perf3 = calc_perf(matched_filter_norm1, ['K=2x3| R=4| ACS=48', 'K=4x5| R=4| ACS=48', 'K=6x7| R=4| ACS=48'],
                     [recon_gg23R4, recon_gg45R4, recon_gg67R4], ssim=False)

    # ...........ACS= 60, kernel = vary................
    g23 = GRAPPA(kdata.copy(), nACS=60, kernel_size=(2, 3))
    g45 = GRAPPA(kdata.copy(), nACS=60, kernel_size=(4, 5))
    g67 = GRAPPA(kdata.copy(), nACS=60, kernel_size=(6, 7))

    gg23R4 = g23.grappa(4, flag_acs=True)
    recon_gg23R4 = ls_comb(ifft2c(gg23R4, axes=(0, 1)), sens_maps, PSI)
    gg45R4 = g45.grappa(4, flag_acs=True)
    recon_gg45R4 = ls_comb(ifft2c(gg45R4, axes=(0, 1)), sens_maps, PSI)
    gg67R4 = g67.grappa(4, flag_acs=True)
    recon_gg67R4 = ls_comb(ifft2c(gg67R4, axes=(0, 1)), sens_maps, PSI)
    imshow1row([recon_gg23R4, recon_gg45R4, recon_gg67R4],
               ['K=2x3| R=4| ACS=60', 'K=4x5| R=4| ACS=60', 'K=6x7| R=4| ACS=60'], filename='3b_60')

    error4 = diff_images(abs(matched_filter_norm1), [recon_gg23R4, recon_gg45R4, recon_gg67R4])
    imshow1row(error2, ['K=2x3| R=4| ACS=60', 'K=4x5| R=4| ACS=60', 'K=6x7| R=4| ACS=60'], filename='e_4')
    perf4 = calc_perf(matched_filter_norm1, ['K=2x3| R=4| ACS=60', 'K=4x5| R=4| ACS=60', 'K=6x7| R=4| ACS=60'],
                     [recon_gg23R4, recon_gg45R4, recon_gg67R4], ssim=False)




