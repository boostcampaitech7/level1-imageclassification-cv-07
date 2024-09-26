import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Callable, Union, Tuple
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, exp, config, info_df: pd.DataFrame, transform: Callable, is_inference: bool = False):
        self.transform = transform
        self.is_inference = is_inference
        self.image_paths = info_df['image_path'].tolist()
        self.exp = exp
        self.use_pseudo_label = 0

        if not self.is_inference:
            self.targets = info_df['target'].tolist()
            self.root_dir = config['train_data_dir']
        else:
            self.root_dir = config['test_data_dir']
        
        if 'pseudo_label' in info_df.columns:
            self.pseudo_dir = config['test_data_dir']
            self.pseudo_label = info_df['pseudo_label'].tolist()
            self.use_pseudo_label = 1


    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        if self.use_pseudo_label and self.pseudo_label[index]:
            img_path = os.path.join(self.pseudo_dir, self.image_paths[index])
        else:
            img_path = os.path.join(self.root_dir, self.image_paths[index])
        
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 흑백 이미지 3차원으로 변환
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)  # (H, W) -> (H, W, 3)
        image = self.transform(image)

        if self.is_inference:
            return image
        else:
            target = self.targets[index]
            return image, target
