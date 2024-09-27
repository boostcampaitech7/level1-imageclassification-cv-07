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
from dotenv import load_dotenv


# 추론(inference) 함수
def inference(model, device, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
    return predictions

# 'best_model' 파일을 찾는 함수 (가장 최근에 저장된 파일 선택)
def get_best_model_path(directory):
    files = [f for f in os.listdir(directory) if f.startswith('best_model') and f.endswith('.pt')]
    if not files:
        raise FileNotFoundError(f"No best model files found in directory: {directory}")
    
    # 파일의 수정 시간을 기준으로 가장 최근에 저장된 파일 선택
    best_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return os.path.join(directory, best_file)

# 디렉터리가 없으면 생성하는 함수
def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main(exp, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로드 (테스트 데이터)
    test_info = pd.read_csv(config['test_info_file'])

    # 변환 설정 (val_transform 사용)
    transform_selector = TransformSelector(transform_type="albumentations")
    test_transform = transform_selector.get_transform(is_train=False)

    # 테스트 데이터셋 및 데이터로더 생성
    test_dataset = CustomDataset(exp, config, info_df=test_info, transform=test_transform, is_inference=True)
    test_loader = DataLoader(test_dataset, batch_size=exp['batch_size'], shuffle=False)

    # 모델 설정
    model_selector = ModelSelector(model_type="timm", num_classes=exp['num_classes'], model_name=exp['model_name'], pretrained=False)
    model = model_selector.get_model()

    # 베스트 모델 경로 설정
    model_path = get_best_model_path(exp['result_path'])
    print(f"Loading best model from {model_path}")

    # 저장된 모델 로드
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 추론 실행
    predictions = inference(model, device, test_loader)

    # 결과 저장 디렉터리 확인 및 생성
    output_dir = os.path.dirname(exp['output_path'])
    ensure_dir_exists(output_dir)

    # 예측 결과 저장
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv(exp['output_path'], index=False)
    print(f"Predictions saved to {exp['output_path']}")
