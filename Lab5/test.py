import scipy.io
import numpy as np
import utils_dist as helper
from utils_dist import paddedzoom
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mat = scipy.io.loadmat('data_brain_8coils.mat')
    kdata = mat['d']
    sens_maps = mat['c']
    noise_maps = mat['n']

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

    R = [1, 2, 3, 4]
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

    helper.imshow1row(recon)
    helper.imshow1row(gmap)

    ls_comb_w_n_norm2 = ls_comb_w_n / ls_comb_w_n.max()
    ls_comb_w_n_norm1 = (ls_comb_w_n - ls_comb_w_n.mean()) / (ls_comb_w_n.std())
    helper.imshow1row([ls_comb_w_n_norm1, ls_comb_w_n_norm2])
    error = helper.diff_images(abs(ls_comb_w_n_norm1), recon)
    helper.imshow1row(error)

   # perf = helper.calc_perf(ls_comb_w_n_norm1, title, recon)

