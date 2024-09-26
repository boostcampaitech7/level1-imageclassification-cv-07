import os
import json
import torch
import pandas as pd
from datasets.custom_dataset import CustomDataset
from datasets.transform import TransformSelector
from models.model_selector import ModelSelector
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from torch.nn.functional import softmax
from datasets.transform import AlbumentationsTransform
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# inference.py
# 기본 변환 (Resize, Normalize, ToTensor만 적용)
def get_basic_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.865, 0.865, 0.865], std=[0.26, 0.26, 0.26]),
        ToTensorV2()
    ])

# 증강 1 (간단한 HorizontalFlip 및 Rotate)
def get_augmentation1_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.Resize(224, 224),
        A.Normalize(mean=[0.865, 0.865, 0.865], std=[0.26, 0.26, 0.26]),
        ToTensorV2()
    ])

# 증강 2 (ElasticTransform 및 GaussNoise 추가)
def get_augmentation2_transform():
    return A.Compose([
        A.ElasticTransform(alpha=1, sigma=10, alpha_affine=10, p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.Resize(224, 224),
        A.Normalize(mean=[0.865, 0.865, 0.865], std=[0.26, 0.26, 0.26]),
        ToTensorV2()
    ])
    
# 추론(inference) 함수 with TTA (Test Time Augmentation)
def inference(model, device, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            outputs = model(images)
            probabilities = softmax(outputs, dim=1)
            preds = torch.argmax(probabilities, dim=1)
            predictions.extend(preds.cpu().numpy())
    return predictions

# 디렉터리가 없으면 생성하는 함수
def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로드 (테스트 데이터)
    test_info = pd.read_csv(config['test_info_file'])

    # 변환 설정 (val_transform 사용)
    transform_selector = TransformSelector(transform_type="albumentations")
    test_transform = transform_selector.get_transform(is_train=False)

    # TTA transform 설정 (TTA용 변환)
    #tta_transform = get_tta_transform()

    # 테스트 데이터셋 및 데이터로더 생성
    test_dataset = CustomDataset(root_dir=config['test_data_dir'], info_df=test_info, transform=test_transform, is_inference=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # 모델 설정
    model_selector = ModelSelector(model_type="timm", num_classes=config['num_classes'], model_name=config['model_name'], pretrained=False)
    model = model_selector.get_model()

    # 베스트 모델 경로 설정
    model_path = '/data/ephemeral/home/Sojeong/level1-imageclassification-cv-07/Other/Sojeong/0925TTA/best_model_1.0949.pt'
    print(f"Loading best model from {model_path}")

    # 저장된 모델 로드
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # # TTA 적용을 위한 결과 누적 변수
    # tta_steps = config.get('tta_steps', 5)
    # aggregated_predictions = np.zeros((len(test_info), config['num_classes']))

    # for _ in range(tta_steps):
    #     # TTA 변환 적용
    #     test_dataset.transform = tta_transform

    #     # 추론 실행 (단일 추론)
    #     predictions = inference(model, device, test_loader)

    #     # 각 추론 결과를 누적
    #     for i, pred in enumerate(predictions):
    #         aggregated_predictions[i, pred] += 1
        # TTA 적용을 위한 결과 누적 변수
    aggregated_predictions = np.zeros((len(test_info), config['num_classes']))

    # 1. 기본 변환 적용 후 추론
    test_dataset.transform = get_basic_transform()
    predictions = inference(model, device, test_loader)
    for i, pred in enumerate(predictions):
        aggregated_predictions[i, pred] += 1

    # 2. 증강 1 적용 후 추론
    test_dataset.transform = get_augmentation1_transform()
    predictions = inference(model, device, test_loader)
    for i, pred in enumerate(predictions):
        aggregated_predictions[i, pred] += 1

    # 3. 증강 2 적용 후 추론
    test_dataset.transform = get_augmentation2_transform()
    predictions = inference(model, device, test_loader)
    for i, pred in enumerate(predictions):
        aggregated_predictions[i, pred] += 1

    # 최종 예측: 각 클래스의 누적값을 기반으로 가장 높은 값을 예측으로 선택
    final_predictions = np.argmax(aggregated_predictions, axis=1)
        
    # 결과 저장 디렉터리 확인 및 생성
    output_dir = os.path.dirname(config['output_path'])
    ensure_dir_exists(output_dir)

    # 예측 결과 저장
    test_info['target'] = final_predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv(config['output_path'], index=False)
    print(f"Predictions saved to {config['output_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference using pre-trained model")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config)
