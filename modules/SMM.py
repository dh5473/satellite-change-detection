from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import (Conv2d, InstanceNorm2d, Module, PReLU,
                      Sequential, Upsample,GELU)
from .ffc_modules import FourierBlock

##################################################################################################
#                                      Fourier Block                                             #
##################################################################################################

class ScaleMixingModule(Module):
    '''
    Mixing image features with different scale by using Fourier Block

    Reference : https://github.com/likyoo/open-cd/blob/main/opencd/models/decode_heads/changer.py#L67
    '''
    
    def __init__(
        self,
        scale_factor: int,
        ch_in: int,
        ch_out: int,
        decode_channel: int
    ):
        super().__init__()

        self._upsample = Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )

        self._conv = Sequential(
            Conv2d((ch_in + decode_channel)//2, ch_out, stride=1, kernel_size=1, padding=0, bias=True),
            InstanceNorm2d(ch_in//2),
            GELU()
        )

        self.flow_ffc = Sequential(
            FourierBlock(1,ch_in + decode_channel,ch_in + decode_channel, return_all=True)
        ) 

    def forward(self, x: Tensor, y: Tensor) -> Tensor:

        x = self._upsample(x)
        output = torch.cat([x, y], dim=1)  
        output,output_loc,output_glo = self.flow_ffc(output)
        fusion = torch.abs(output_glo - output_loc)  
        fusion = self._conv(fusion)

        return fusion


##################################################################################################
#                                         UpMask                                                 #
##################################################################################################
    
class UpMask(Module):
    def __init__(
        self,
        scale_factor: float,
        nin: int,
        nout: int,
    ):
        super().__init__()
        self._upsample = Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )
        self._convolution = Sequential(
            Conv2d(nin, nin, 3, 1, groups=nin, padding=1),
            PReLU(),
            InstanceNorm2d(nin),
            Conv2d(nin, nout, kernel_size=1, stride=1),
            PReLU(),
            InstanceNorm2d(nout),
        )

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        x = self._upsample(x)
        if y is not None:
            x = x * y
        return self._convolution(x)