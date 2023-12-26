import numpy as np
import torch
from PIL import Image
import os

def transform(input_image, imtype=np.uint8):
    image_numpy = input_image
    if image_numpy.shape[0] == 1: 
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0 
    return image_numpy.astype(imtype)

def save_images(img_name, img_numpy, save_path):
    os.makedirs(save_path,exist_ok=True)
    img_path = os.path.join(save_path, img_name)
    image_pil = Image.fromarray(img_numpy)
    image_pil.save(img_path)

 