from metrics.metric_tool import ConfuseMatrixMeter
from dataset.inference_dataset import InferenceDataset
from dataset.dataset import MyDataset
from torch.utils.data import DataLoader
from models.tinycd import TinyCD
from utils.utils import save_images, transform
import torch
import tqdm
import os
import numpy as np

class DHJModel:
    def __init__(self,mode):
        data_path = os.path.join('data/LEVIR-CD',mode)
        model_path = 'pretrained_models/model_19.pth'
        
        self.save_path = os.path.join('results',mode)
        self.tool_metric = ConfuseMatrixMeter(n_class=2)
        self.criterion = torch.nn.BCELoss()
        self.bce_loss = 0

        self.dataset = self.load_dataset(data_path,mode)
        self.data_loader = DataLoader(self.dataset, batch_size=1)
        
        self.device = self.device_assign()
        self.model = TinyCD()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.to(self.device)

    def device_assign(self):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        return device
    
    def load_dataset(self,data_path,mode):
        dataset = MyDataset(data_path, mode) if mode == 'test' else InferenceDataset(data_path, mode)
        return dataset

    def inference(self):
        with torch.no_grad():
            for reference, testimg, img_name in tqdm.tqdm(self.data_loader):
                reference = reference.to(self.device).float()
                testimg = testimg.to(self.device).float()

                generated_mask = self.model(reference, testimg).squeeze(1)
                bin_genmask = (generated_mask.to("cpu") >0.5).numpy().astype(int)

                bin_genmask = transform(bin_genmask)
                save_images(img_name[0], bin_genmask, self.save_path)

    def test(self):
        with torch.no_grad():
            for (reference, testimg), mask, img_name in tqdm.tqdm(self.data_loader):
                reference = reference.to(self.device).float()
                testimg = testimg.to(self.device).float()
                mask = mask.float()

                generated_mask = self.model(reference, testimg).squeeze(1)
                bin_genmask = (generated_mask.to("cpu") >0.5).numpy().astype(int)
                self.bce_loss += self.criterion(generated_mask, mask)
  
                mask = mask.numpy()
                mask = mask.astype(int)
                self.tool_metric.update_cm(pr=bin_genmask, gt=mask)

                bin_genmask = transform(bin_genmask)
                save_images(img_name[0], bin_genmask, self.save_path)

            self.bce_loss /= len(self.data_loader)
            print("Test summary")
            print("Loss is {}".format(self.bce_loss))
            scores_dictionary = self.tool_metric.get_scores()
            print(scores_dictionary)

if __name__ == "__main__":
    model = DHJModel("inference")
    model.inference()