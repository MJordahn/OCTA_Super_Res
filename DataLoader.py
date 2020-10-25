import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from albumentations import Compose

class MabulaDataset(Dataset):

    def __init__(self, file_path, transforms=None):
        self.image_list = os.listdir(os.getcwd() + file_path)
        self.root_dir = os.getcwd()+file_path
        #self.input_lines = input_lines
        if transforms != None:
            self.transforms = Compose(transforms)
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.image_list[idx])

        image = cv2.imread(img_name, 0)
        image = image.astype("uint8")
        image = image.transpose()
        if self.transforms:
            image = self.transforms(image = image)
        image['image'] = torch.unsqueeze(image['image'], 0)

        return image
