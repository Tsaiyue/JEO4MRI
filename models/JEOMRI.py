import torch
import torch.nn as nn
import transforms_simple as T
import pdb

from torch.nn import functional as F
from utils import *
from SWT import SWTForward, SWTInverse


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int = 2,
        out_chans: int = 2,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob 


        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
     
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

        
    def forward(self, image):
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
        # residual connection
        output = output + image
        return output

    

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)



class myAtA(nn.Module):
    """
    performs DC step
    """
    def __init__(self, csm, mask):
        super(myAtA, self).__init__()
        self.csm = csm # complex (B x ncoil x nrow x ncol)
        self.mask = mask # complex (B x nrow x ncol)

    def forward(self, xk): #step for batch image
        xk = xk.permute(0, 2, 3, 1)
        xk_coil = T.expand_operator(xk, self.csm, dim=1)
        k_full = T.fft2(xk_coil, shift=False)
        k_und = self.mask * k_full
        x_u_coil = T.ifft2(k_und, shift=False)
        output = T.reduce_operator(x_u_coil, self.csm, dim=1)
        output = output.permute(0, 3, 1, 2)
        return output

class myWtW(nn.Module):
    """
    performs DC step
    """
    def __init__(self, ock):
        super(myWtW, self).__init__()
        # replacing full one in ll channels of ock
        b, h, w = ock.shape[0], ock.shape[-2], ock.shape[-1]
        full_ones = torch.ones(size=(b, 1, h, w)).to(device=ock.device)
        ock = torch.cat((full_ones, ock[:, 0:3, ...], full_ones, ock[:, 3:6, ...]), dim=1)
        self.ock = ock # complex (B x ncoil x nrow x ncol)

    def forward(self, xk): #step for batch image
        sfm = SWTForward()
        ifm = SWTInverse()

        coeffs = sfm(xk)
        icoeffs = (self.ock ** 2) * coeffs
        output = ifm(icoeffs)


        return output
    
class gradient_descend(nn.Module):
    
    def __init__(self):
        super(gradient_descend, self).__init__()
        self.step = nn.Parameter(torch.tensor(0.5))
        self.rho = nn.Parameter(torch.tensor(0.5))  
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, xk, x0, mask, csm, ock, z):
    
        AtA = myAtA(csm, mask)
        WtW = myWtW(ock)
        dc_res = AtA(xk) - x0
        edge_res = WtW(xk)
        reg_res = z - xk
        output = xk - self.step * (dc_res + self.rho * edge_res- self.beta * reg_res)

        return output

class close_form(nn.Module):
    
    def __init__(self):
        super(close_form, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.rho = nn.Parameter(torch.tensor(0.5))
        

    def forward(self, v, xk):
       
        sfm = SWTForward(xk) 
        Wx = sfm(xk)
        Wx = torch.cat((Wx[:, 1:4, ...], Wx[:, 5:8, ...]), dim=1)
        Wx2 = Wx ** 2
        output = self.alpha * v / (self.rho * Wx2 + self.alpha)

        return output
    
class JEOMRI(nn.Module):
    
    def __init__(self, cascades=5):
        super(JEOMRI, self).__init__()
        
        self.cascades = cascades 
        denoiser_blocks = []
        gd_blocks = []
        recovery_blocks = []
        cf_blocks = []

        
        for i in range(cascades):
        # for i in range(1):
            denoiser_blocks.append(Unet(in_chans=2, out_chans=2, drop_prob=0.05)) 
            gd_blocks.append(gradient_descend()) 
            recovery_blocks.append(Unet(in_chans=6, out_chans=6, drop_prob=0.05))
            cf_blocks.append(close_form())
        
        self.denoiser_blocks = nn.ModuleList(denoiser_blocks)
        self.gd_blocks = nn.ModuleList(gd_blocks)
        self.recovery_blocks = nn.ModuleList(recovery_blocks)
        self.cf_blocks = nn.ModuleList(cf_blocks)
        

    def forward(self, x0, mask, csm, oc0):
        xk = x0.clone()
        ock = oc0.clone()
        for i in range(self.cascades):
            v = self.recovery_blocks[i](ock)
            ock = self.cf_blocks[i](v, xk)
            z = self.denoiser_blocks[i](xk)
            xk = self.gd_blocks[i](xk, x0, mask, csm, ock, z)
        
        return xk, ock 
