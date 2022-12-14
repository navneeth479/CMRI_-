import scipy.io


if __name__ == "__main__":
    mat = scipy.io.loadmat('data_brain_8coils.mat')
    kdata = mat['d']
    sens_maps = mat['c']
    noise_maps = mat['n']