import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image

class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    实时读取图片版 (On-the-fly loading)，不依赖大容量 HDF5。
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Load encoded captions
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)
            
        # 加载图片路径列表
        with open(os.path.join(data_folder, self.split + '_IMGPATHS_' + data_name + '.json'), 'r') as j:
            self.img_paths = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # 1. 获取图片路径
        path = self.img_paths[i]
        
        # 2. 实时读取图片 (PIL)
        # 确保转为 RGB，防止灰度图或CMYK报错
        img = Image.open(path).convert('RGB')
        
        # 3. 应用预处理 (Resize -> Tensor -> Normalize)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all captions
            all_captions = torch.LongTensor([self.captions[i]])
            
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size