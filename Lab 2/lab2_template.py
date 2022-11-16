import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from skimage.filters import window
import matplotlib.colors as clr

if __name__ == "__main__":
    # load matlab file
    mat = scipy.io.loadmat('kdata1.mat')['kdata1']
    mat2 = scipy.io.loadmat('kdata2.mat')['kdata2']

    plt.imshow(np.log(np.abs(mat)), cmap='gray')
    plt.show()

    recon = np.sqrt(np.size(mat))*np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mat)))
    #plt.imshow(np.absolute(recon), cmap='gray')
    #plt.show()
    #plt.imshow(np.angle(mat),vmin=-np.pi,vmax=np.pi); plt.colorbar()
    #plt.show()

    recon2 = np.fft.ifft2(mat)
    #plt.imshow(np.absolute(recon2), cmap='gray')
    #plt.show()

# 2 take a central k space line, create a box, then for image create a Mask and add mask to k-space and see effects
kline = mat[256]

length = kline.shape[0]
box = np.zeros_like(kline)
plt.plot(np.abs(kline))
plt.show()

frame = np.zeros_like(mat)
mask = np.ones([128, 128])
frame_size = frame.shape[0]


recon = mask * mat

# 3 PSF apply ifft on box car func. Then apply it to mask for all sizes
plt.plot(box)
plt.show()

psf = np.fft.ifft2c(box[..., None]).squeeze(1)
f,a = plt.subplot(1,2)

signal_len = len(psf)
psf_shift = psf - np.max(psf/2)
minarg = np.argmin(np.abs(psf_shift))
fwhm = minarg - signal_len//2 if minarg > signal_len//2 else signal_len // 2 - minarg
fwhm = fwhm * 2

# 4 Windowing (filters)

Hamming = filters.H









