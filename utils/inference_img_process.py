from PIL import Image
import torchvision.transforms as transforms



import torch
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import models
import torch.nn as nn
from torch.autograd import Function
import os
import shutil


def img_preprocess(filename):
    image = Image.open('.' + filename)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    preprocess = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      normalize
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_tensor = preprocess(image)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    return img_tensor