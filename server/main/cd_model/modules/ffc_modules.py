import torch
import copy
from torch.nn import (Conv2d, Module, ModuleList, PReLU,
                      Sequential, BatchNorm2d, ReLU)
from typing import List
from torch import Tensor

##################################################################################################
#                                      Fourier Block                                             #
##################################################################################################

class Spectra(Module):
    def __init__(self, in_depth,AF='prelu'):
        super().__init__()
        
        #Params
        self.in_depth = in_depth
        self.inter_depth = self.in_depth//2 if in_depth>=2 else self.in_depth

        #Layers
        self.AF1 = ReLU if AF=='relu' else PReLU(self.inter_depth)
        self.AF2 = ReLU if AF=='relu' else PReLU(self.inter_depth)
        self.inConv = Sequential(Conv2d(self.in_depth,self.inter_depth,1),
                                    BatchNorm2d(self.inter_depth),
                                    self.AF1)
        self.midConv = Sequential(Conv2d(self.inter_depth,self.inter_depth,1),
                                    BatchNorm2d(self.inter_depth),
                                    self.AF2)
        self.outConv = Conv2d(self.inter_depth, self.in_depth, 1)
        
    def forward(self,x):
        x = self.inConv(x)
        skip = copy.copy(x)
        rfft = torch.fft.rfft2(x)  
        real_rfft = torch.real(rfft)  
        imag_rfft = torch.imag(rfft)  
        cat_rfft = torch.cat((real_rfft,imag_rfft),dim=-1)  
        cat_rfft = self.midConv(cat_rfft)
        mid = cat_rfft.shape[-1]//2  
        real_rfft = cat_rfft[...,:mid]
        imag_rfft = cat_rfft[...,mid:]
        rfft = torch.complex(real_rfft,imag_rfft)  #
        spect = torch.fft.irfft2(rfft)
        out = self.outConv(spect + skip)  
        return out
    

class FastFC(Module):
    def __init__(self,in_depth,out_depth,AF='prelu'):
        super().__init__()
        #Params
        self.in_depth = in_depth//2
        self.out_depth = out_depth
        
        #Layers
        self.AF1 = ReLU if AF=='relu' else PReLU(self.in_depth)
        self.AF2 = ReLU if AF=='relu' else PReLU(self.in_depth)
        self.conv_ll = Conv2d(self.in_depth,self.in_depth,3,padding='same')
        self.conv_lg = Conv2d(self.in_depth,self.in_depth,3,padding='same')
        self.conv_gl = Conv2d(self.in_depth,self.in_depth,3,padding='same')
        self.conv_gg = Spectra(self.in_depth, AF)
        self.bnaf1 = Sequential(BatchNorm2d(self.in_depth),self.AF1)
        self.bnaf2 = Sequential(BatchNorm2d(self.in_depth),self.AF2)
        self.conv_final = Conv2d(self.in_depth*2,self.out_depth,3,padding='same')
        
    def forward(self,x):
        mid = x.shape[1]//2
        x_loc = x[:,:mid,:,:]
        if x.shape[1]%2 != 0:
            x_glo = x[:,mid+1:,:,:]
        else:
            x_glo = x[:,mid:,:,:]

        x_ll = self.conv_ll(x_loc)
        x_lg = self.conv_lg(x_loc)
        x_gl = self.conv_gl(x_glo)
        x_gg = self.conv_gg(x_glo)
        out_loc = torch.add((self.bnaf1(x_ll + x_gl)),x_loc)
        out_glo = torch.add((self.bnaf2(x_gg + x_lg)),x_glo)
        out = torch.cat((out_loc,out_glo),dim=1)
        out = self.conv_final(out)
        return out,out_loc,out_glo

    
class FourierBlock(Module):
    def __init__(self,num_layer,in_depth,out_depth,return_all=False, attention=False):
        super().__init__()
        #Params
        self.num_layers = num_layer
        self.in_depth = in_depth
        self.out_depth = out_depth
        self.return_all  = return_all
        #Layers
        self.block = ModuleList()
        for _ in range(self.num_layers):
            self.block.append(FastFC(self.in_depth,self.out_depth,'prelu'))

    def forward(self,x):  
        for layer in self.block:
            x,x_loc,x_glo = layer(x)
        if self.return_all:
            return x,x_loc,x_glo
        else:
            return x
        

##################################################################################################
#                                     PixelWise Linear                                           #      
##################################################################################################
        
class PixelwiseLinear(Module):
    def __init__(
        self,
        fin: List[int],
        fout: List[int],
        first_mix,
        last_activation: Module = None,
    ) -> None:
        assert len(fout) == len(fin)
        super().__init__()

        n = len(fin)
        self._linears = Sequential(
            *[
                Sequential(
                    FourierBlock(1, fin[i],fout[i]) if first_mix == False else Conv2d(fin[i], fout[i], kernel_size=1, bias=True),
                    PReLU()
                    if i < n - 1 or last_activation is None
                    else last_activation,
                )
                for i in range(n)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._linears(x)