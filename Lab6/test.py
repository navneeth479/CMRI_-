import scipy.io
import numpy as np
import utils_dist as helper
from utils_dist import *
from utils_dist import paddedzoom
import matplotlib.pyplot as plt
from grappa_dist import GRAPPA

if __name__ == "__main__":
    mat = scipy.io.loadmat('data_brain_8coils.mat')
    kdata = mat['d']  # (PE,RO,nCoil)
    sens_maps = mat['c']  # (PE,RO,nCoil)
    noise_maps = mat['n']  # (RO,nCoil)
    PSI = np.cov(noise_maps, rowvar=False)  # (nCoil, nCoil)

    coil_recon = helper.ifft2c(kdata, axes=(0, 1))
    coil_recon_t = coil_recon.transpose(2, 0, 1)

    coil_recon_sum = np.sum(coil_recon, axis=2)
    helper.imshow1row([coil_recon_sum])

    sos = helper.sos_comb(coil_recon)
    PSI = np.cov(noise_maps.T)
    helper.imshow1row([PSI])
    ls_comb_wo_n = helper.ls_comb(coil_recon, sens_maps)
    ls_comb_w_n = helper.ls_comb(coil_recon, sens_maps, PSI)
    helper.imshow1row([coil_recon_sum, sos, ls_comb_wo_n, ls_comb_w_n],
                      ['Complex sum', 'SoS', 'LS w/o PSI', 'LS w/ PSI'], isMag=True)

    R = [2, 3, 4]
    title = []
    recon = []
    gmap = []
    for i in range(len(R)):
        ia = helper.ifft2c(kdata[::R[i]], axes=(0, 1))
        ir, g = helper.sense_recon(ia, sens_maps, PSI, R[i])
        norm_ir = abs((ir-ir.mean())/(ir.std()))
        recon.append(norm_ir)
        gmap.append(g)
        title.append('R=' + str(R[i]))

    helper.imshow1row(recon, title, filename='SENSE')
    helper.imshow1row(gmap)

    ls_comb_w_n_norm1 = (ls_comb_w_n - ls_comb_w_n.mean()) / (ls_comb_w_n.std())

    perf = helper.calc_perf(ls_comb_w_n_norm1, title, recon)

    # ....Experiment (3.b)
    recon = ifft2c(kdata, axes=(0, 1))
    matched_filter = ls_comb(recon, sens_maps, PSI)
    matched_filter_norm1 = (matched_filter - matched_filter.mean()) / (matched_filter.std())
    # ...........ACS= 24, kernel = vary................
    g23 = GRAPPA(kdata.copy(), nACS=24, kernel_size=(2, 3))

    gg23R2 = g23.grappa(2, flag_acs=True)
    recon_gg23R4 = ls_comb(ifft2c(gg23R2, axes=(0, 1)), sens_maps, PSI)
    gg23R3 = g23.grappa(3, flag_acs=True)
    recon_gg45R4 = ls_comb(ifft2c(gg23R3, axes=(0, 1)), sens_maps, PSI)
    gg23R4 = g23.grappa(4, flag_acs=True)
    recon_gg67R4 = ls_comb(ifft2c(gg23R4, axes=(0, 1)), sens_maps, PSI)
    imshow1row([recon_gg23R4, recon_gg45R4, recon_gg67R4],
               ['R=2', 'R=3', 'R=4'], filename='GRAPPA')

    perf1 = calc_perf(matched_filter_norm1, ['R=2', 'R=3', 'R=4'],
                      [recon_gg23R4, recon_gg45R4, recon_gg67R4], ssim=False)

