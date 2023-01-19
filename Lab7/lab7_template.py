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

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    # 2.1 print trajectory
    plt.plot(traj[:, :].real, traj[:, :].imag)
    plt.title('k-space trajectory (64 spokes)')
    plt.show()

    kspace_T = kspace.transpose(2,0,1)
    sens_maps_t = sens_maps.transpose(2,0,1)
    imshow1row([kspace_T[1,:,:], sens_maps_t[1,:,:]], titles=['Radial kdata (channel 1)',  'rx coil sensitivity map (channel 1)'], isMag=True, norm=0.2)

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
    k_dcomp = torch.tensor(k_dcomp.transpose(2, 0, 1).reshape(kspace.shape[-1], -1)[None,...], device=DEVICE)

    # nufft recon results
    nufft_recon = adjnufft_ob(k_dcomp, traj_torch, smaps=smaps)
    imshow1row([gt, torch.squeeze(nufft_recon).cpu().numpy()], titles=['GT','NUFFT reconstruction'])

    # Grad decent algorithm

    t = 3E-2
    u = torch.zeros(nufft_recon.shape, dtype=nufft_recon.dtype, device = DEVICE)
    k = torch.zeros(u.shape, dtype=u.dtype, device=DEVICE)
    kdata = torch.tensor(kspace.transpose(2,0,1).reshape(kspace.shape[-1], -1)[None, ...], device=DEVICE)
    iterations = 300
    recons = []
    kdata_evolve = []
    mse = []
    l2 = []
    error_loss = []
    with tqdm(total=iterations, unit='iter', leave=True, colour='green') as pbar:
        for i in range(iterations):
            grad = 2 * adjnufft_ob(
                                   nufft_ob(u, traj_torch, smaps=smaps, norm='ortho') - kdata,
                                   traj_torch,
                                    smaps=smaps,
                                    norm='ortho')
            u = u - t * grad
            k = ifft2c(torch.squeeze(u).cpu().numpy())
            np.power(k, 0.3, k)

            pbar.set_description(desc=f'Iteration {i:3d}')
            pbar.update()
            recons.append(abs(torch.squeeze(u).cpu().numpy()))
            kdata_evolve.append(abs(k))

            MSE = np.square(np.subtract(abs(normalize_img(gt)), abs(normalize_img(recons[len(recons) - 1])))).mean()
            mse.append(MSE)
            l2_loss = np.sqrt(np.sum(np.square(abs(normalize_img(gt)) - abs(normalize_img(recons[len(recons) - 1])))))
            error_loss.append(0.01*l2_loss)
            L2_grad = grad.norm(p=2)
            l2.append(L2_grad*0.1)


    creat_gif(recons, 'GD 300-Iterations', duration=50)
    creat_gif(kdata_evolve, 'GD k-data 300-Iterations', duration=50,)
    imshow1row(recons[-1:], titles=[f'GD recon at iteration {iterations}'])
    imshow1row(kdata_evolve[-1:], titles=[f'GD k-space at iteration {iterations}'], isMag=True)
    imshow1row([gt, torch.squeeze(nufft_recon).cpu().numpy(),recons[len(recons) - 1]], titles=['GT', 'Regridded', 'GD iterations=300'])
    e = diff_images(abs(normalize_img(gt)), [abs(normalize_img(torch.squeeze(nufft_recon).cpu().numpy())),abs(normalize_img(recons[len(recons) - 1]))])
    imshow1row(e, titles=['Regridded error', 'GD iterations=300 error'])
    plt.plot(range(0, iterations), mse)
    plt.plot(range(0, iterations), l2)
    #plt.plot(range(0, iterations), error_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Metric values')
    plt.title('GD algorithm convergence')
    plt.legend(['mse', '0.1 * (L2 norm)'])
    plt.grid()
    plt.show()





















