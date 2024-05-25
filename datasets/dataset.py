import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
import h5py as h5
import numpy as np
import cv2
from utils import c2r
import matplotlib.pyplot as plt
from SWT import SWTForward, SWTInverse




############################################
class dataset(Dataset):
    def __init__(self, mode, dataset_path, sigma=0.01):
        """
        :sigma: std of Gaussian noise to be added in the k-space
        """
        self.prefix = 'trn' if mode == 'train' else 'tst'
     
        ##########################################
        self.dataset_path = dataset_path
        self.sigma = sigma

    def __getitem__(self, index):
        """
        :x0: zero-filled reconstruction (2 x nrow x ncol) - float32
        :gt: fully-sampled image (2 x nrow x ncol) - float32 range of x0 & gt: (-1, 1)
        :csm: coil sensitivity map (ncoil x nrow x ncol) - complex64
        :mask: undersample mask (nrow x ncol) - int8
        """
        
        with h5.File(self.dataset_path, 'r') as f:
            gt, csm, mask = f[self.prefix+'Org'][index], f[self.prefix+'Csm'][index], f[self.prefix+'Mask'][index]
        x0, k_und = undersample(gt, csm, mask, self.sigma)
        # mask = np.stack([mask, mask])
        # return torch.from_numpy(c2r(x0)), torch.from_numpy(c2r(gt)), torch.from_numpy(c2r(csm)), torch.from_numpy(mask)
        x0 = torch.from_numpy(c2r(x0))
        gt = torch.from_numpy(c2r(gt))
        k_und = torch.from_numpy(k_und)
        csm = torch.from_numpy(csm)
        mask = torch.from_numpy(mask)        
        
        ########## testing the wavelet transform opt ##########
        
        sfm = SWTForward()
        ifm = SWTInverse()

        x0 = x0.unsqueeze(0)
        gt = gt.unsqueeze(0)

        # wavelet transform dec
        coeffs_x0 = sfm(x0)
        coeffs_gt = sfm(gt)

        # get rid of ll in real & imagnary part
        coeffs_x0 = torch.cat([coeffs_x0[:, 1:4, :, :], coeffs_x0[:, 5:8, :, :]], dim=1)
        coeffs_gt = torch.cat([coeffs_gt[:, 1:4, :, :], coeffs_gt[:, 5:8, :, :]], dim=1)

        # abs, normalize & inverse
        coeffs_x0 = torch.abs(coeffs_x0)
        coeffs_gt = torch.abs(coeffs_gt)

        max_coeffs_x0 = coeffs_x0.max()
        min_coeffs_x0 = coeffs_x0.min()
        gap_coeffs_x0 = max_coeffs_x0 - min_coeffs_x0
        coeffs_x0 = (coeffs_x0 - min_coeffs_x0) / gap_coeffs_x0

        max_coeffs_gt = coeffs_gt.max()
        min_coeffs_gt = coeffs_gt.min()
        gap_coeffs_gt = max_coeffs_gt - min_coeffs_gt
        coeffs_gt = (coeffs_gt - min_coeffs_gt) / gap_coeffs_gt

        coeffs_x0 = 1.0 - coeffs_x0
        coeffs_gt = 1.0 - coeffs_gt
        
        coeffs_x0 = coeffs_x0.squeeze(0)
        coeffs_gt = coeffs_gt.squeeze(0)

        x0 = x0.squeeze(0)
        gt = gt.squeeze(0)


            
        return x0, gt, csm, mask, coeffs_x0, coeffs_gt

    def __len__(self):
        with h5.File(self.dataset_path, 'r') as f:
            num_data = len(f[self.prefix+'Mask'])
        return num_data

def undersample(gt, csm, mask, sigma):
    """
    :get fully-sampled image, undersample in k-space and convert back to image domain
    """
    ncoil, nrow, ncol = csm.shape
    sample_idx = np.where(mask.flatten()!=0)[0]
    noise = np.random.randn(len(sample_idx)*ncoil) + 1j*np.random.randn(len(sample_idx)*ncoil)
    noise = noise * (sigma / np.sqrt(2.))
    k_und_flatten, k_und_imageshape = piA(gt, csm, mask, nrow, ncol, ncoil)
    b = k_und_flatten + noise #forward model
    atb = piAt(b, csm, mask, nrow, ncol, ncoil)
    return atb, k_und_imageshape

def piA(im, csm, mask, nrow, ncol, ncoil):
    """
    fully-sampled image -> undersampled k-space
    """
    im = np.reshape(im, (nrow, ncol))
    im_coil = np.tile(im, [ncoil, 1, 1]) * csm #split coil images
    k_full = np.fft.fft2(im_coil, norm='ortho') #fft
    
    if len(mask.shape) == 2:
        mask = np.tile(mask, (ncoil, 1, 1))

    k_u_imageshape = k_full * mask
    k_u_flatten = k_full[mask!=0]
    
    return k_u_flatten, k_u_imageshape 

def piAt(b, csm, mask, nrow, ncol, ncoil):
    """
    k-space -> zero-filled reconstruction
    """
    if len(mask.shape) == 2:
        mask = np.tile(mask, (ncoil, 1, 1))
    zero_filled = np.zeros((ncoil, nrow, ncol), dtype=np.complex64)
    zero_filled[mask!=0] = b #zero-filling
    img = np.fft.ifft2(zero_filled, norm='ortho') #ifft
    coil_combine = np.sum(img*csm.conj(), axis=0).astype(np.complex64) #coil combine
    return coil_combine
