from typing import List, Tuple
from collections import Sized
from os.path import join
import albumentations as alb
from torchvision.transforms import Normalize

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
        self._A = join(data_path, "A")
        self._B = join(data_path, "B")
        self._list_images = self._read_images_list(data_path)

        # Initialize normalization:
        self._normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    def __getitem__(self, indx):
        # Current image set name:
        img_name = self._list_images[indx].strip('\n')

        # Loading the images:
        x_ref = imread(join(self._A, img_name))
        x_test = imread(join(self._B, img_name))

        # Trasform data from HWC to CWH:
        x_ref, x_test = self._to_tensors(x_ref, x_test)

        return x_ref, x_test,img_name

    def __len__(self):
        return len(self._list_images)

    def _read_images_list(self, data_path: str) -> List[str]:
        images_list_file = join(data_path,'list', self._mode + ".txt")
        with open(images_list_file, "r") as f:
            return f.readlines()
    
    def _to_tensors(
        self, x_ref: np.ndarray, x_test: np.ndarray
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return (
            self._normalize(torch.tensor(x_ref).permute(2, 0, 1)),
            self._normalize(torch.tensor(x_test).permute(2, 0, 1)),
        )