from .metrics.metric_tool import ConfuseMatrixMeter
from .dataset.inference_dataset import InferenceDataset
from .dataset.dataset import MyDataset
from torch.utils.data import DataLoader
from .models.tinycd import TinyCD
from .utils.utils import transform, pad_and_crop, restore_imgs
from PIL import Image
from matplotlib.image import imread

import numpy as np
import torch
import tqdm
import os
import datetime


class DHJModel:
    def __init__(self, mode):
        model_path = 'main/cd_model/outputs/best_weights/TinyCD/levir.pth'
        
        self.mode = mode
        self.base_path = 'main/cd_model/data/INFERENCE-CD'
        self.save_path = 'static/result'

        self.tool_metric = ConfuseMatrixMeter(n_class=2)
        self.criterion = torch.nn.BCELoss()
        self.bce_loss = 0

        self.device = self.device_assign()
        self.model = TinyCD()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.model.to(self.device)

    def device_assign(self):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        return device

    def load_dataset(self, data_path,mode):
        dataset = MyDataset(data_path, mode) if mode == 'test' else InferenceDataset(data_path, mode)
        return dataset

    def inference(self, A_, B_):
        # 추론 폴더 생성
        now = datetime.datetime.now()
        folder_name = f"{'inference'}{now.strftime('%Y-%m-%d_%H-%M-%S')}"
        folder_path = os.path.join(self.base_path, folder_name)
        A_path = os.path.join(folder_path, 'inference', 'A')
        B_path = os.path.join(folder_path, 'inference', 'B')

        try:
            os.makedirs(A_path, exist_ok=True)
            os.makedirs(B_path, exist_ok=True)
            print(f"폴더 '{folder_name}'가 생성되었습니다.")
        except OSError as e:
            print(f"폴더 생성 실패: {e}")        

        # 이미지 padding & crop 후 저장
        A = imread(A_)
        B = imread(B_)
        x, y, As = pad_and_crop(A, (256, 256))
        _, _, Bs = pad_and_crop(B, (256, 256))

        for i in range(len(As)):
            img_path_A = os.path.join(A_path, str(i) + '.png')
            img_path_B = os.path.join(B_path, str(i) + '.png')

            image_pil_A = Image.fromarray((As[i] * 255).astype(np.uint8))
            image_pil_B = Image.fromarray((Bs[i] * 255).astype(np.uint8))

            image_pil_A.save(img_path_A)
            image_pil_B.save(img_path_B)

        dataset = self.load_dataset(folder_path, self.mode)
        data_loader = DataLoader(dataset, batch_size=1)

        save_path = os.path.join(self.save_path, folder_name)
        results = []

        with torch.no_grad():
            for reference, testimg, _ in tqdm.tqdm(data_loader):

                reference = reference.to(self.device).float()
                testimg = testimg.to(self.device).float()

                generated_mask = self.model(reference, testimg).squeeze(1)
                generated_mask = generated_mask.to("cpu")

                bin_genmask = (generated_mask > 0.5).numpy().astype(int)
                bin_genmask = transform(bin_genmask)

                results.append(bin_genmask)

        result_img_pil = restore_imgs(results, x, y, A.shape[0], A.shape[1])
        os.makedirs(save_path, exist_ok=True)
        result_img_pil.save(os.path.join(save_path, 'result.png'))


# if __name__ == "__main__":
#     model = DHJModel("inference")
#     A = "C:\\projects\\dhj\\static\\prepare\\A\\20240102_154739\\response.tiff"
#     B = "C:\\projects\\dhj\\static\\prepare\\B\\20240102_154753\\response.tiff"
#     model.inference(A, B)