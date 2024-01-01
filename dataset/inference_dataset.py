from typing import List, Tuple
from collections import Sized
from os.path import join
import os
from torchvision.transforms import Normalize, Compose

import numpy as np
import torch
from matplotlib.image import imread
from torch.utils.data import Dataset
from torch import Tensor


class InferenceDataset(Dataset, Sized):
    def __init__(
        self,
        data_path: str,
        mode: str
    ) -> None:

        self._mode = mode
        self._A = join(data_path,self._mode ,"A")
        self._B = join(data_path,self._mode, "B")
        self._list_images = self._read_images_list(data_path)

        # Initialize normalization:
        self._normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        self._preprocess = Compose([self._normalize])
        
    def __getitem__(self, indx):
        # Current image set name:
        img_name = self._list_images[indx]

        # Loading the images:
        x_ref = imread(join(self._A, img_name))
        x_test = imread(join(self._B, img_name))

        # Trasform data from HWC to CWH:
        x_ref, x_test = self._to_tensors(x_ref, x_test)

        return x_ref, x_test,img_name

    def __len__(self):
        return len(self._list_images)

    def _read_images_list(self, data_path: str) -> List[str]:
        data_path = os.path.join(data_path, 'inference', 'A')
        img_list = os.listdir(data_path)
        return img_list
    
    def _to_tensors(
        self, x_ref: np.ndarray, x_test: np.ndarray
    ) -> Tuple[Tensor, Tensor]:
        
        x_ref = self._preprocess(torch.tensor(x_ref).permute(2, 0, 1))
        x_test = self._preprocess(torch.tensor(x_test).permute(2, 0, 1))
        
        return (
            x_ref,
            x_test
        )