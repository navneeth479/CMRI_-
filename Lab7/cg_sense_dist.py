'''
CG-SENSE iterative reconstruction

Original author:    Bruno Riemenschneider
First created:      03.01.2022

Modifier:
Email:
Last update:
'''
import torch
import torchkbnufft as tkbn
import numpy as np
from typing import Optional
from tqdm.auto import tqdm
from utils_dist import creat_gif, ifft2c, normalize_img


def cg_sense(gt: np.ndarray,
             data: np.ndarray,
             traj: np.ndarray,
             sens: np.ndarray,
             maxit: Optional[int] = 50,
             tol: Optional[float] = 1e-6,
             device=torch.device("cpu"),
             title: Optional[str] = None
             ) -> np.ndarray:
    '''
    Reconstruct subsampled PMRI data using CG SENSE [1]
    uses M. Muckley's torchkbnufft: https://github.com/mmuckley/torchkbnufft

    INPUT
    @data:      numpy 3D array [[kspace defined by traj] x coils] of coil data
    @traj:      trajectory, 2D array [read x lines] of complex trajectory points in [-0.5,0.5]
    @sens:      numpy 3D array of coil sensitivities [read x phase x coils]
    @maxit:     maximum number of CG iterations
    @device:    device where calculation is performed
    @title:     if it's given, recon images are saved in a gif over all iterations

    OUTPUT:
             reconstructed image

    Create:         03.01.2022
    By:             Bruno Riemenschneider
    Last Change:    14.12.2022
    By:             Jinho Kim

    [1] Pruessmann, K. P.; Weiger, M.; Boernert, P. and Boesiger, P.
    Advances in sensitivity encoding with arbitrary k-space trajectories.
    Magn Reson Med 46: 638-651 (2001)

    =========================================================================
    '''

    kdata = torch.tensor(data.transpose(2,0,1).reshape(data.shape[-1], -1)[None, ...], device=device)
    smaps = torch.tensor(sens.transpose(2,0,1)[None, ...], device=device)
    traj *= 2*np.pi
    im_size = sens.shape[:-1]
    grid_size = (sens.shape[0], sens.shape[0])
    traj_stack = torch.tensor(np.stack((traj.real.flatten(), traj.imag.flatten()), axis=0), device=device)

    nufft_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size, device=device)
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size, device=device)

    x = torch.zeros(sum(((1, 1), im_size), ()), device=device)
    r = adjnufft_ob(kdata, traj_stack, smaps=smaps, norm='ortho')
    p = r
    rzold = torch.real(my_dot(r, r))
    mse = []

    gifs = []
    with tqdm(total=maxit, unit='iter', leave=True) as pbar:
        for ii in range(maxit):
            Ap = adjnufft_ob(
                nufft_ob(p, traj_stack, smaps=smaps, norm='ortho'),
                traj_stack,
                smaps=smaps,
                norm='ortho')
            alpha = rzold / torch.real(my_dot(p, Ap))
            x = x + alpha * p
            r = r - alpha * Ap
            rznew = torch.real(my_dot(r, r))
            if rznew < tol:
                break

            beta = rznew / rzold
            p = r + beta * p
            rzold = rznew

            pbar.set_description(desc=f"Iteration {ii: 3d}")
            pbar.set_postfix({"residual norm": None})
            pbar.update()
            # residual does not take regularization into account
            k = nufft_ob(x, traj_stack, smaps=smaps, norm='ortho')
            gifs.append(x)
            #gifs.append(k)
            MSE = np.square(np.subtract(abs(normalize_img(gt)), abs(normalize_img(np.squeeze(x.cpu().numpy()))))).mean()
            mse.append(MSE)

    if title:
        gifs = [abs(i[0, 0].cpu().numpy()) for i in gifs]
        creat_gif(gifs, title)

    return mse, np.squeeze(x.cpu().numpy())


def my_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.vdot(torch.flatten(a), torch.flatten(b))
