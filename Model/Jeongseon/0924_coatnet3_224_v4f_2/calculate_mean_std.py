import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets.custom_dataset import CustomDataset
from datasets.transform import TransformSelector
import json
import argparse
import torch

# 데이터셋의 평균과 표준편차 계산
def calculate_mean_std(root_dir, info_df, transform):
    dataset = CustomDataset(root_dir, info_df, transform, is_inference=True)
    
    channel_sum = np.zeros(3)
    channel_sum_squared = np.zeros(3)
    pixel_count = 0

    for idx in tqdm(range(len(dataset)), desc="Calculating mean and std"):
        image = dataset[idx]  #이미지 불러옴
        
        # CustomDataset에서 이미 변환된 이미지를 반환한다고 가정
        # 만약 torch.Tensor라면 numpy로 변환
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        # 채널 순서가 (C, H, W)라면 (H, W, C)로 변경
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        #image = image / 255.0
        
        channel_sum += np.sum(image, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(image), axis=(0, 1))
        pixel_count += image.shape[0] * image.shape[1]

    channel_mean = channel_sum / pixel_count
    channel_std = np.sqrt(channel_sum_squared / pixel_count - np.square(channel_mean))

    return channel_mean, channel_std

def main(config):
    # 데이터 로드
    train_info = pd.read_csv(config['data_info_file'])
    
    # train_index.csv 파일 경로
    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    rel_train_index_path = os.path.normpath("datasets/train_index.csv")
    train_index_path = os.path.join(py_dir_path, rel_train_index_path)
    
    # train_index.csv를 이용하여 train_df를 로드
    train_index = pd.read_csv(train_index_path, header=None).squeeze()
    train_df = train_info.loc[train_index]

    # 변환 설정 (기본 변환만 적용)
    transform_selector = TransformSelector(transform_type="albumentations")
    transform = transform_selector.get_transform(is_train=False)

    # 평균과 표준편차 계산
    mean, std = calculate_mean_std(config['train_data_dir'], train_df, transform)

    print("Calculated mean:", mean)
    print("Calculated std:", std)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate dataset mean and std")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config)