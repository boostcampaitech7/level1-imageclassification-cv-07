import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Callable, Union, Tuple
import numpy as np


def check_file_exists(file_path):
    """
    주어진 경로에 파일이 존재하는지 확인합니다.
    
    :param file_path: 확인할 파일의 경로
    :return: 파일이 존재하면 True, 그렇지 않으면 False
    """
    return os.path.isfile(file_path)


class CustomDataset(Dataset):
    def __init__(self, root_dir: str, info_df: pd.DataFrame, transform: Callable, is_inference: bool = False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_inference = is_inference
        self.image_paths = info_df['image_path'].tolist()

        if not self.is_inference:
            self.targets = info_df['target'].tolist()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        
        img_path = os.path.join(self.root_dir, self.image_paths[index])
        if check_file_exists(img_path):
            print(f"파일이 존재합니다: {img_path}")
        else:
            print(f"파일이 존재하지 않습니다: {img_path}")
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
