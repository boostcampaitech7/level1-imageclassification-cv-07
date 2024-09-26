import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Callable, Union, Tuple
import numpy as np
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir: str, info_df: pd.DataFrame, transform: Callable, is_inference: bool = False):
        self.root_dir = root_dir            #이미지 파일이 저장된 디렉토리 경로
        self.transform = transform          #이미지 경로 & 타겟정보 들어있는 df
        self.is_inference = is_inference    #추론여부
        self.image_paths = info_df['image_path'].tolist()

        if not self.is_inference:
            self.targets = info_df['target'].tolist()

    def __len__(self) -> int:               #데이터셋의 총 데이터 수 반환
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]: #인덱스에 해당하는 데이터 반환
        img_path = os.path.join(self.root_dir, self.image_paths[index])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 흑백 이미지 3차원으로 변환
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)  # (H, W) -> (H, W, 3)
        #image = image.astype(np.float32) / 255.0  # 픽셀 값 정규화
        image = Image.fromarray(np.uint8(image)) # numpy -> PIL 이미지로 변환
        #print("imageType: ", type(image))
        image = self.transform(image)

        if self.is_inference: #추론이면 이미지만 반환
            return image
        else: #학습이면 이미지와 타겟 반환
            target = self.targets[index]
            return image, target
