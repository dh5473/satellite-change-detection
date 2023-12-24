"""
Codegoni A, Lombardi G, Ferrari A. 
TINYCD: A (Not So) Deep Learning Model For Change Detection[J]. 
arXiv preprint arXiv:2207.13159, 2022.
The code in this file is borrowed from:
https://github.com/AndreaCodegoni/Tiny_model_4_CD
"""
from typing import List

import torchvision
from torch import Tensor
from torch.nn import (Module, ModuleList)

from modules.ffc_modules import PixelwiseLinear
from modules.ESAMB import ESAMB, ScaleMixingModule, FeatureFusion
from modules.TMM import TemporalMixingModule, UpMask


##################################################################################################
#                                     Feature Fusion                                             #
##################################################################################################


def _get_backbone(
    bkbn_name, pretrained, output_layer_bkbn, freeze_backbone) -> ModuleList:
    # The whole model:
    entire_model = torchvision.models.vit_l_16(pretrained=pretrained)

    print(entire_model)
    # Slicing it:
    derived_model = ModuleList([])
    for name, layer in entire_model.named_children():
        derived_model.append(layer)
        if name == '4':
            break

    # Freezing the backbone weights:
    if freeze_backbone:
        for param in derived_model.parameters():
            param.requires_grad = False
    return derived_model


class TinyCD(Module):
    def __init__(
        self,
        bkbn_name="vit_l_16",
        pretrained=True,
        output_layer_bkbn="4",
        freeze_backbone=False,
    ):
        super().__init__()
        
        # Load the pretrained backbone according to parameters:
        self._backbone = _get_backbone(
            bkbn_name, pretrained, output_layer_bkbn, freeze_backbone
        )

        # Initialize mixing blocks:
        self._first_mix = ESAMB(6, 3,first_mix=True)
        self._mixing_mask = ModuleList(
            [
                ESAMB(48, 24),
                ESAMB(64, 32),
                ESAMB(112, 56),
                FeatureFusion(224, 112,'skip_sub'),
            ]
        )

        # Initialize Upsampling blocks:
        self._up = ModuleList(
            [
                TemporalMixingModule(2, 112, 56, 56),
                TemporalMixingModule(2, 56, 64, 32),
                TemporalMixingModule(2, 64, 64, 24),
                TemporalMixingModule(2, 64, 32, 3),
            ]
        )

        # Final classification layer:
        self._classify = PixelwiseLinear([32, 16,8], [16,8, 1], None) # out_channels = 8

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:

        features = self._encode(x1, x2)
        latents = self._decode(features)
        out = self._classify(latents)

        return (out,)

    def _encode(self, ref, test) -> List[Tensor]:

        features = [self._first_mix(ref, test)]
        for num, layer in enumerate(self._backbone):
            ref, test = layer(ref), layer(test)
            if num != 0:
                features.append(self._mixing_mask[num - 1](ref, test))

        return features

    def _decode(self, features) -> Tensor:

        upping = features[-1]
        for i, j in enumerate(range(-2, -6, -1)):
            upping = self._up[i](upping, features[j])

        return upping