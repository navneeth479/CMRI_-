'''
Compressed Sensing reconstruction

author: Jinho Kim
Email: jinho.kim@fau.de
Last update: Dec. 19. 2022
'''
import numpy as np
import scipy.io
from utils_dist import *
import pickle
from mri_utils import generateSamplingMask

mat = scipy.io.loadmat('data_lab8.mat')
kfull = mat['kfull']
kacc = mat['kacc']
kfull = np.flip(kfull)
kacc = np.flip(kacc)

#############################################################################################
# Additional sample masks.....

mask_path='mask_1dg_a5.pickle'  #path for the required mask
maf=open(mask_path,'rb')
mask=pickle.load(maf)
#########################################################################################

# mri utils generates masks ....
# Select an undesampling ratio

delta = 0.35
# Generate an undersampling mask
omega = generateSamplingMask(kfull.shape, delta, 'vardentri')
omega= np.invert(omega)
# Plot mask
plt.imshow(omega[0], cmap='binary')
plt.axis('off')
plt.show()
############################################################################################
print(np.count_nonzero(kfull)/np.count_nonzero(omega[0] * kfull))
##############


gt = ifft2c(kfull)
imshow1row([gt], titles=['Ground truth'])

c_complex_array, s = dwt2(gt)
imshow1row([c_complex_array], titles=['Wavelet coeffs'], norm=0.2)
l1 = calc_perf(gt, [ gt ,c_complex_array], l1= True)

#  Checking for sparsity with l1 norm.......

# Compress and recover......with rmse and error
C = [5, 10, 20]
recons = []
for factor in C:
    compressed = compress(c_complex_array.copy(), factor)
    recon = idwt2(compressed, s)
    recons.append(recon)

rmses = calc_perf(gt, recons, rmse=True)
imshow_metric(recons, titles=[f"Compress_factor: {c}" for c in C], val='RMSE', metrics=rmses, filename='rmse')
imshow1row([gt-i for i in recons], titles=[f"Compress_factor: {c}" for c in C], norm=0.3, filename='error')

# ... To show sparsity.........................
c_acc, s_acc = dwt2(ifft2c(kacc))
c_omega, s_omega = dwt2(abs(fft2c(omega[0] * kfull)))

imshow1row([omega[0] * kfull, abs(fft2c(omega[0] * kfull)), c_omega], titles=['kspace(Gaussian density)', 'Image domain','Wavelet domain (D4)'], norm=0.2, filename=' Gaussian Density')
l1 = calc_perf(gt, [ abs(fft2c(omega[0] * kfull)) ,c_omega], l1= True)

imshow1row([kacc, abs(fft2c(kacc)), c_acc], titles=['Undersampled kspace', 'Image domain', 'Wavelet domain (D4)'], norm=0.2, filename='undersampled')
l1 = calc_perf(gt, [ abs(fft2c(kacc)) ,c_acc], l1= True)


# ... CS_ista results .........with modified masks
lamdas = [5, 1, 0.5]
for i, lamda in enumerate(lamdas, start=1):
    rec, inter_m, cost = cs_ista(omega[0] * kfull, lamda, 30)
    imshow1row([inter_m[0], inter_m[-1], abs(gt-normalize_img(inter_m[-1]))], titles=['Initial', 'CS recovered', 'error against GT'])
    create_gif(inter_m, title=str(lamda), duration=50)
    plt.plot(range(0, 30),cost)
    plt.xlabel('Epoch')
    plt.ylabel('Metric values')
    plt.title('algorithm convergence')
    plt.grid()
    plt.show()




