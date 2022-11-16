import matplotlib.pyplot as plt
import numpy as np
import scipy.io

# load matlab file
mat = scipy.io.loadmat('digital_brain_phantom.mat')
label = mat['ph']['label'][0][0]
T1_map = mat['ph']['t1'][0][0]
T2_map = mat['ph']['t2'][0][0]
SD_map = mat['ph']['sd'][0][0]
(nR, nC) = label.shape

# 1. Inspecting Brain phantom contrasts

CSF_seg = (label == 1)
Grey_seg = (label == 2)
White_seg = (label == 3)
segments = np.dstack((CSF_seg, Grey_seg, White_seg))
labels = ['CSF', 'Grey Matter', 'White Matter']

fig = plt.figure(figsize=(7, 7))
for i in np.arange(np.shape(segments)[2]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(segments[:, :, i], cmap='gray')
    plt.title(labels[i])
    plt.axis("off")
fig.tight_layout()
plt.show()



# We assume PD is equal to M0.
# Predefined T1, T2 & PD values for CSF, GM & WM

CSF_ind = np.argwhere(CSF_seg)
Grey_ind = np.argwhere(Grey_seg)
White_ind = np.argwhere(White_seg)

CSF_vals = GM_vals = WM_vals = []

T1_CSF, T2_CSF, PD_CSF = ['CSF T1:', T1_map[CSF_ind[0, 0], CSF_ind[0, 1]]], \
                         ['CSF T2:', T2_map[CSF_ind[0, 0], CSF_ind[0, 1]]], \
                         ['CSF PD:', SD_map[CSF_ind[0, 0], CSF_ind[0, 1]]]
T1_Grey, T2_Grey, PD_Grey = ['GM T1:', T1_map[Grey_ind[0, 0], Grey_ind[0, 1]]],\
                            ['GM T2:', T2_map[Grey_ind[0, 0], Grey_ind[0, 1]]],\
                            ['GM PD:', SD_map[Grey_ind[0, 0], Grey_ind[0, 1]]]
T1_White, T2_White, PD_White = ['WM T1:', T1_map[White_ind[0, 0], White_ind[0, 1]]],\
                               ['WM T2:', T2_map[White_ind[0, 0], White_ind[0, 1]]],\
                               ['WM PD:', SD_map[White_ind[0, 0], White_ind[0, 1]]]

CSF_vals = [T1_CSF, T2_CSF, PD_CSF]
GM_vals = [T1_Grey, T2_Grey, PD_Grey]
WM_vals = [T1_White, T2_White, PD_White]

vals = np.hstack((CSF_vals, GM_vals, WM_vals))
print(vals)


# 2.1 See the lecture for the values discussed in PDw for TE and TR
# 2.1.2 (vary the TR; TE for PDw, T1w, T2w)
TE = [15, 15, 120]  # order PDw, T1_w, T2_w
TR = [3000, 500, 5000]
PDw = SD_map * (1 - 2 * np.exp(-(TR[0] - TE[0]) / T1_map) + np.exp(-TR[0] / T1_map)) * np.exp(-TE[0] / T2_map)
# PDw_GM = SD_map * (1 - 2 * np.exp(-(TR - TE) / T1_Grey[1]) + np.exp(-TR / T1_Grey[1])) * np.exp(-TE / T2_Grey[1])
# PDw_WM = SD_map * (1 - 2 * np.exp(-(TR - TE) / T1_White[1]) + np.exp(-TR / T1_White[1])) * np.exp(-TE / T2_White[1])

T1_w = SD_map * (1 - 2 * np.exp(-(TR[1] - TE[1]) / T1_map) + np.exp(-TR[1] / T1_map)) * np.exp(-TE[1] / T2_map)
T2_w = SD_map * (1 - 2 * np.exp(-(TR[2] - TE[2]) / T1_map) + np.exp(-TR[2] / T1_map)) * np.exp(-TE[2] / T2_map)
w = np.dstack((PDw, T1_w, T2_w))
title = ['PDw', 'T1w', 'T2w']
fig = plt.figure(figsize=(7, 7))
for i in np.arange(np.shape(w)[2]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(w[:, :, i], cmap='gray')
    plt.title(title[i] + " (TR=" + str(TR[i]) + " TE=" + str(TE[i])+")",
              fontsize='large',
              loc='left',
              style='italic',
              family='monospace')
    plt.axis("off")
fig.tight_layout()
plt.show()


# 2.2 FLAIR sequence
# Water suppression needs large TR = 10 sec (TI for nulling CSF)
TR = 10000
TI = T1_CSF[1] * (np.log(2) - np.log(1 + np.exp(-TR / T1_CSF[1])))
TE = 100
'180 pulse for inversion for flair'
theta = np.pi
FLAIR_seq = (SD_map * (1 - (1 - np.cos(theta)) * np.exp(-TI / T1_map) + np.exp(-TR / T1_map))) * np.exp((-TE/T2_map))
FLAIR_seq_1 = (SD_map * (1 - (1 - np.cos(theta)) * np.exp(-TI / T1_map) + np.exp(-TR / T1_map)))
title = ['Flair', 'Flair with TE']
mat = np.dstack((FLAIR_seq_1, FLAIR_seq))
fig = plt.figure(figsize=(7, 7))
for i in np.arange(np.shape(mat)[2]):
    plt.subplot(1, 2, i + 1)
    plt.imshow(mat[:, :, i], cmap='gray')
    plt.title(title[i] ,
              fontsize='large',
              loc='center',
              style='italic',
              family='monospace')
    plt.axis("off")
fig.tight_layout()
plt.show()


# 2.3 MPRAGE sequence
TR = 4300
TE = 3.68
TI = 910
alpha = 12 * (np.pi / 180)
ES = 20.2
B1 = 1  # assume to be 1 since phantom is single slice
RP = 1  # assume to be 1 since phantom is single slice
n1 = 80
N3d = nC
tau1 = TI - ES * n1
tau2 = TR - tau1 - ES * N3d

T1_star = np.power((1 / T1_map) - (1 / ES) * np.log(np.cos(alpha * B1)), -1)

E1 = np.exp(-tau1 / T1_map)
E2 = np.exp(-tau2 / T1_map)
E3 = np.exp(-(N3d * ES) / T1_star)
E4 = np.exp(-(n1 * ES) / T1_star)

Q_numerator1 = E4 * (1 - 2 * E1 + E1 * E2)
Q_numerator2 = T1_star / T1_map * (1 + E1 * E2 * E3 - E1 * E2 * E4 - E4)
Q_denominator = 1 + E1 * E2 * E3
Q = (Q_numerator1 + Q_numerator2) / Q_denominator

'MPRAGE sequence signal'
S = SD_map * np.sin(alpha * B1) * RP * Q

plt.imshow(S, cmap='gray')
plt.title("MPRAGE")
plt.axis("off")
plt.show()

