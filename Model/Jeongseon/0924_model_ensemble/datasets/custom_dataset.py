import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Callable, Union, Tuple
import numpy as np

# CustomDataset 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, root_dir: str, info_df: pd.DataFrame, transform: Callable, is_inference: bool = False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_inference = is_inference
        self.image_paths = info_df['image_path'].tolist()  #Dataframe에서 image_path column을 list로 변환

        if not self.is_inference:
            self.targets = info_df['target'].tolist()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]: #함수가 반환할 수 있는 타입을 명시(이미지텐서, 레이블)or 이미지 텐서
        img_path = os.path.join(self.root_dir, self.image_paths[index]) #이미지 경로(루트디렉토리 + 이미지경로)
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
