import numpy as np
import scipy.io
from grid import grid
import torchkbnufft as tkbn
import torch
import utils_dist
import matplotlib.pyplot as plt

if __name__ == "__main__":
    k_radial = scipy.io.loadmat('radial_data.mat')['k']

    '''
    nRO: the number of readout points
    nSpocks: the number of kdata trajectories.
    '''
    nRO, nSpokes = k_radial.shape
    # 1. Plot trajectory
    PI = np.pi
    GA = 111.246117975 * (PI/180)
    'Normalize spoke length from -.5 to .5 cause of gridding func'
    r = np.linspace(-0.5, 0.5, nRO).reshape((nRO,1))  # K-space trajectory
    print(r.shape)

    theta = np.mod(np.arange(PI/2, nSpokes * GA, GA), 2*PI)
    print(theta.min(), theta.max(), theta.shape)

    traj = r * np.exp(1j*theta)
    print(traj.shape)

    ns = 20
    plt.plot(traj[:,:ns].real, traj[:,:ns].imag)
    plt.show()
    '....... Do the spokes match Nyquist rate? ........'

    # 2 Basic gridding reconstruction
    k_cartesian = grid(k_radial, k=traj, n=nRO)
    print(k_cartesian.shape)

    recon = utils_dist.ifft2c(k_cartesian)
    utils_dist.imshow1row([k_cartesian], ["k-space"],  isMag=True, norm=0.1)
    utils_dist.imshow1row([abs(recon)], ["Image"], isMag=False) # But blurry cause of additional projections acquired over full rotation
    utils_dist.imshow1row([k_radial, k_cartesian], isMag=True, norm=0.1)

    #3 Compensate density using ramp
    def get_ramp(k_radial):
        nRO , nSpocks = k_radial.shape
        ramp = np.abs(np.linspace(-1 + 1/nSpocks, 1-1/nSpocks, nRO))[:, None] + 1/nSpocks
        return ramp


    ramp = get_ramp(k_radial)
    k_radial_ramp = ramp * k_radial
    k_cart_ramp = grid(k_radial_ramp, traj, nRO)
    k_cart_ramp_recon = utils_dist.ifft2c(k_cart_ramp)
    utils_dist.imshow1row([abs(k_cart_ramp_recon), abs(recon)], ['Ramp filtered', 'Original'], isMag=False)

    #4 Oversampling
    os_rate = 1.5
    k_cart_ramp_os = grid(k_radial_ramp, traj, int(nRO * os_rate))
    k_cart_ramp_os_recon = utils_dist.ifft2c(k_cart_ramp_os)
    k_cart_ramp_os_recon_cropped = utils_dist.crop(k_cart_ramp_os_recon, int(np.multiply(nRO, os_rate)), os_rate)
    utils_dist.imshow1row([abs(k_cart_ramp_os_recon), abs(k_cart_ramp_recon), abs(recon)],
                          ['over sampled', 'Ramp filtered', 'Original'], isMag=False)









