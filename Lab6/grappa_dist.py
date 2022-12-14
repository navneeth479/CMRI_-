'''
GRAPPA reconstruction, which can use flexible kernel sizes and acceleration rates
Based on Ricardo Otazo's MATLAB implementation

Author: Jinho Kim
Email: jinho.kim@fau.de
First created: Tue. Dec. 06. 2022
Last modified: Tue. Dec. 07. 2022
'''
import numpy as np


# todo: Implement each methods based on the method descriptions.
class GRAPPA:
    def __init__(self, kspace, nACS=24, kernel_size=(2, 3)):
        '''
        Initialize GRAPPA object.
        @param kspace: fully-sampled kspace
        @param nACS: The number of ACS lines (default: 24)
        @param kernel_size: The kernel size (default: 2x3)
        '''
        self.ws = None
        self.kspace = kspace
        self.nACS = nACS
        self.PE, self.RO, self.nCoil = kspace.shape
        self.kernel_PE, self.kernel_RO=kernel_size


    def _set_params(self, R):
        '''
        Depending on the acceleration factor, R, set parameters differently.
        @param R: The acceleration factor
        '''
        self.R = R
        self.zf_kspace = self._undersample()
        self.acs = self._get_acs()

        self.block_w = self.kernel_RO
        self.block_h = self.R * (self.kernel_PE - 1) + 1

        self.nb = (self.nACS - self.block_h + 1) * (self.RO - self.block_w + 1)
        self.nkc = self.kernel_RO * self.kernel_PE * self.nCoil



    def _get_acs(self):
        '''
        Get ACS from the kspace
        @return: ACS
        '''
        # todo
        acs_u = self.PE//2 - self.nACS//2
        acs_d = self.PE//2 + self.nACS//2
        return self.kspace[acs_u:acs_d]

    def _undersample(self):
        '''
        Undersample the kspace while keeping the original size
        @return: Undersampled kspace
        '''
        mask = self._get_mask()
        undersampled = self.kspace * mask
        return undersampled

    def _get_mask(self):
        '''
        Get undersampling mask, which keeps the original size
        @return: Undersampling mask
        '''
        # todo
        mask = np.zeros_like(self.kspace)
        mask[::self.R] = 1
        return mask

    def _extract(self):
        '''
        Extract source and target points from ACS
        @return:
            src : source points (shape: nb x nck)
            targ: target points (shape: R-1 x nb x nc)
        '''
        src = np.zeros((self.nb, self.nkc), dtype=self.kspace.dtype)
        target = np.zeros((self.R-1, self.nb, self.nCoil), dtype=self.kspace.dtype)
        box_idx = 0

        for idx_RO in range(self.kernel_RO // 2, self.RO - self.kernel_RO // 2):
            for idx_ACS in range(self.nACS - self.block_h + 1):
                src[box_idx] = np.array(
                    [
                        [
                            self.acs[idx_ACS + dy * self.R, idx_RO + dx] for dy in range(self.kernel_PE)
                        ] for dx in range(-(self.kernel_RO // 2), self.kernel_RO // 2 + 1)
                    ]
                ).flatten()

                for dy in range(self.R - 1):
                    target[dy, box_idx] = np.array(
                        self.acs[idx_ACS + (self.kernel_PE // 2 - 1) * self.R + dy + 1, idx_RO]
                    ).flatten()
                box_idx += 1

        return src, target

    def _interpolation(self, zp_kspace):
        '''
        Interpolate missing points (zeros in the kspace)
        @param zp_kspace: Zero-padded undersampled kspace
        @return:
            interpolated: interpolated kspace
        '''
        interpolated = zp_kspace.copy()
        for idx_RO in range(self.kernel_RO // 2, self.RO - self.kernel_RO // 2):
            for idx_PE in range(0, self.PE - self.block_h+1, self.R):
                source = np.array(
                    [
                        [
                            zp_kspace[idx_PE + self.R * dy, idx_RO + dx] for dy in range(self.kernel_PE)
                        ] for dx in range(-(self.kernel_RO // 2), self.kernel_RO//2 + 1)
                    ]
                ).flatten()

                for dy in range(self.R - 1):
                    interpolated[idx_PE + (self.kernel_PE // 2 - 1)*self.R + dy + 1, idx_RO] = np.dot(source, self.ws[dy])

        return interpolated

    def _zero_padding(self):
        '''
        Zero padding to make sure every skipped line been filed
        @return:
            zp_kdata: zero-padded kspace
        '''
        zp_kdata = np.zeros((self.PE + self.R * 2, self.RO + self.kernel_RO // 2 * 2, self.nCoil), dtype=self.kspace.dtype)
        zp_kdata[self.R : self.PE + self.R, self.kernel_RO // 2: self.RO + self.kernel_RO//2, :] = self.zf_kspace
        return zp_kdata

    def _crop2original(self, interpolated):
        '''
        Crop the padded kspace to its original size
        @param interpolated: interpolated kspace
        @return: cropped kspace
        '''
        return interpolated[self.R : self.PE + self.R, self.kernel_RO // 2: self.RO + self.kernel_RO//2, :]

    def grappa(self, R, flag_acs=False):
        '''
        Core method to perform GRAPPA reconstruction.
        This method performs GRAPPA reconstruction based on given the acceleration factor R
        @param R: The acceleration factor
        @param flag_acs: Whether to keep ACS or not
        @return: GRAPPA reconstructed kspace
        '''
        self._set_params(R)
        src, targ = self._extract()

        # todo
        self.ws = np.linalg.pinv(src)[None, ...] @ targ

        zp_kdata = self._zero_padding()

        interpolated = self._interpolation(zp_kdata)
        interpolated = self._crop2original(interpolated)

        if flag_acs:
            acs_u = self.PE // 2 - self.nACS // 2
            acs_d = self.PE // 2 + self.nACS // 2
            interpolated[acs_u:acs_d] = self.acs

        return interpolated
