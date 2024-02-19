import numpy as np
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

# Pad and Crop 
def pad_and_crop(original_array : np.ndarray, crop_size : (256, 256)):

    # 원본 배열에서 높이, 너비 추출
    original_height, original_width, _ = original_array.shape

    # crop_size에 맞게 패딩
    x = (original_width // 256 + 1) 
    y = (original_height // 256 + 1) 
    padded_array = np.pad(original_array, ((0,y * 256-original_height),(0,x * 256-original_width), (0,0)), 'constant', constant_values=0)

    # crop_size에 따라 자르기
    cropped_arrays = []
    for i in range(original_height // crop_size[0]+ 1):
        for j in range(original_width // crop_size[1]+ 1):
            start_h, start_w = i * crop_size[0], j * crop_size[1]
            end_h, end_w = start_h + crop_size[0], start_w + crop_size[1]
            cropped_array = padded_array[start_h:end_h, start_w:end_w, :]
            cropped_arrays.append(cropped_array)

    return x, y, cropped_arrays


# Restoring image
def restore_imgs(results: list, x: int, y: int, org_shape_x, org_shape_y):
    result_imgs = np.zeros((256*y, 256*x, 3))

    for i in range(y):
        for j in range(x):
            result = results[x*(i) + j]
            result_imgs[256*(i):256*(i+1), 256*(j):256*(j+1),:] = result

    result_imgs = result_imgs[:org_shape_x, :org_shape_y, :]
    result_img_pil = Image.fromarray(result_imgs.astype(np.uint8))

    return result_img_pil 
 