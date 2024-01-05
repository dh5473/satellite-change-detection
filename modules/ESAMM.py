import torch
from torch import Tensor, reshape, stack
from torch.nn import (Conv2d, InstanceNorm2d, Module,PReLU, Sequential)
import torch.nn.functional as F


##################################################################################################
#                                            ESAMB                                               #
##################################################################################################
    
class ESAMM(Module):

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        first_mix: bool = False,
    ):
        self.first_mix = first_mix
        super().__init__()
        self._featurefusion = TemporalMixingModule(ch_in, ch_out,'skip_sub') 
        self._tmm = EfficientSelfAttention(ch_out, ch_out, first_mix)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        z_mix = self._featurefusion(x, y)
        output = self._tmm(z_mix) 

        return output


##################################################################################################
#                                    Efficient Self Attention                                    #
##################################################################################################

class EfficientSelfAttention(Module):
    '''
    Extracting meaningful information from mixed image features by using Efficient-Attention Method

    Reference : 
    https://github.com/cmsflash/efficient-attention/blob/master/efficient_attention.py
    '''

    def __init__(
        self,
        nin: int,
        nout:int,
        first_mix: bool = False
    ):
        super().__init__()

        self.in_channels = nin
        self.key_channels = (nin+1) // 4 if first_mix == False else 2
        self.head_count = (nin+1) // 4 if first_mix == False else 2
        self.value_channels = (nin+1) // 4 if first_mix == False else 2

        self.keys = Conv2d(nin, self.key_channels, 1)
        self.queries = Conv2d(nin, self.key_channels, 1)
        self.values = Conv2d(nin, self.value_channels, 1)
        self.reprojection = Conv2d(self.value_channels, nin, 1)


    def forward(self, x: Tensor) -> Tensor:
        '''
        Multihead cross attention by applying efficient attention method

        x : Feature needs to be upsampled
        y : Attention Mask with higher scale
        '''
        n, _ , h, w = x.size()
        keys = self.keys(x).reshape((n, self.key_channels, h * w))
        queries = self.queries(x).reshape(n, self.key_channels, h * w)
        values = self.values(x).reshape((n, self.value_channels, h * w))        
        head_key_channels = self.key_channels // self.head_count if self.key_channels >= self.head_count else 1
        head_value_channels = self.value_channels // self.head_count if self.key_channels >= self.head_count else 1


        # Multihead attention
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            query = F.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + x

        return attention
    

##################################################################################################
#                                     TMM(Temporal Mixing Module)                                #
##################################################################################################
    
class TemporalMixingModule(Module):
    '''
    Temporally Mixing image features at t1 and t2

    '''
    def __init__(
        self,
        ch_in: int,
        ch_out : int,
        f_method : str
    ):
        self.f_method = f_method
        super().__init__()
        self._convmix = Sequential(
            Conv2d(ch_in//2, ch_out, 3, padding=1),
            PReLU(),
            InstanceNorm2d(ch_out),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:

        if self.f_method == 'stack_reshape':
            mixed = stack((x, y), dim=2)
            mixed = reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))
        elif self.f_method == 'sub':
            mixed = x - y
        elif self.f_method == 'add':
            mixed = x + y
        elif self.f_method == 'mul':
            mixed = x * y
        elif self.f_method == 'abs_sub':
            mixed = torch.abs(x - y)
        elif self.f_method == 'skip_sub':
            sub = x - y
            skip = x * y
            mixed = sub + skip
        result = self._convmix(mixed)

        return result