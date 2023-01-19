'''
CG-SENSE reconstruction

Original author: Bruno Riemenschneider
First created: Jan. 3. 2022

Modifier: Jinho Kim
Email: jinho.kim@fau.de
Last update: Dec. 7. 2022
'''
import numpy as np
import scipy.io
import torch
import torchkbnufft as tkbn
from tqdm.auto import tqdm

from cg_sense_dist import cg_sense
from utils_dist import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    mat = scipy.io.loadmat('data_radial_brain_4ch.mat')
    kspace = mat['kdata']
    sens_maps = mat['c']
    traj = mat['k']  # range of traj: -0.5~0.5
    ds_comp = mat['w']
    gt = mat['img_senscomb']
    N_ro, N_spokes, N_ch = kspace.shape

    # 2.2 Simple regridding operation with NUFFT operator

    im_size = gt.shape
    grid_size = (N_ro, N_ro)
    # set nufft objects
    nufft_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size, device=DEVICE)
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size, device=DEVICE)

    # convert params to tensor objects
    traj_nufft = traj * 2 * np.pi
    traj_torch = torch.tensor(np.stack((traj_nufft.real.flatten(), traj_nufft.imag.flatten()), axis=0), device=DEVICE)
    smaps = torch.tensor(sens_maps.transpose(2, 0, 1)[None, ...], device=DEVICE)
    k_dcomp = kspace.copy()
    k_dcomp = kspace * ds_comp[..., None]
    k_dcomp = torch.tensor(k_dcomp.transpose(2, 0, 1).reshape(kspace.shape[-1], -1)[None, ...], device=DEVICE)

    # nufft recon results
    nufft_recon = adjnufft_ob(k_dcomp, traj_torch, smaps=smaps)
    imshow1row([gt, torch.squeeze(nufft_recon).cpu().numpy()], titles=['GT', 'NUFFT reconstruction'])

    #  CGD reconstruction results .........
    tol = 1e-6
    maxit = 30
    mse, recon_CG = cg_sense(gt.copy(), kspace.copy(), traj.copy(), sens_maps, tol=tol, maxit=maxit, device=DEVICE, title=f"CGD reconstruction")
    k_evolve = ifft2c(recon_CG)

    #  Noise estimation ..........
    #z = 1e-3 * np.random.randn(kspace.shape[0], kspace.shape[1], kspace.shape[2]).astype(dtype=kspace.dtype)
    #kspace = kspace + z
    #noise_CG = cg_sense(gt.copy(), z, traj.copy(), sens_maps, tol=tol, maxit=500, device=DEVICE,
    #                    title=f"k space noise")

    imshow1row([recon_CG], titles=['CG SENSE recon'], isMag=True)
    imshow1row([k_evolve], titles=['CG SENSE kspace'], isMag=True, norm=0.2)
    #imshow1row([noise_CG], titles=['CG SENSE noise estimation (iters=500)'], isMag=True)
    #imshow1row([ifft2c(noise_CG)], titles=['CG SENSE noise k-space(iters=500)'], isMag=True, norm=0.3)
    imshow1row([gt, torch.squeeze(nufft_recon).cpu().numpy(), recon_CG],
               titles=['GT', 'NUFFT', 'CG SENSE'])
    e = diff_images(abs(normalize_img(gt)), [abs(normalize_img(torch.squeeze(nufft_recon).cpu().numpy())),
                                             abs(normalize_img(recon_CG))])
    imshow1row(e, titles=['NUFFT error', 'CG SENSE error'])

    plt.plot(range(0, len(mse)), mse)
    #plt.plot(range(0, iterations), error_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Metric values')
    plt.title('CG SENSE convergence')
    plt.legend(['mse'])
    plt.grid()
    plt.show()




