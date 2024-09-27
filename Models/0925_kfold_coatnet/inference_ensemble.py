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
import glob
import numpy as np

# 추론(inference) 함수
def inference(model, device, test_loader):
    model.eval()
    predictions = [] 
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.softmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    predictions = np.array(predictions)
    return predictions

# 'best_model' 파일을 찾는 함수 (가장 최근에 저장된 파일 선택)
def get_best_model_path(directory):
    files = [f for f in os.listdir(directory) if f.startswith('best_model') and f.endswith('.pt')]
    if not files:
        raise FileNotFoundError(f"No best model files found in directory: {directory}")
    
    # 파일의 수정 시간을 기준으로 가장 최근에 저장된 파일 선택
    best_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return os.path.join(directory, best_file)

def get_exp_json(directory):
    foldname = os.path.split(directory)[1]
    path = os.path.join(directory, f"{foldname}.json")

    with open(path, 'r') as f:
        exp = json.load(f)

    return exp

# 디렉터리가 없으면 생성하는 함수
def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model using config file")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    exp_file_path = args.config

    with open(exp_file_path, 'r') as f:
        config = json.load(f)

    load_dotenv() # load .env file
    config['train_data_dir'] = os.environ.get('train_data_dir')
    config['test_data_dir'] = os.environ.get('test_data_dir')
    config['data_info_file'] = os.environ.get('data_info_file')
    config['test_info_file'] = os.environ.get('test_info_file')

    # 계산 시작
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로드 (테스트 데이터)
    test_info = pd.read_csv(config['test_info_file'])

    # 변환 설정 (val_transform 사용)
    transform_selector = TransformSelector(transform_type="albumentations")
    test_transform = transform_selector.get_transform(is_train=False)

    # 테스트 데이터셋 및 데이터로더 생성
    test_dataset = CustomDataset(root_dir=config['test_data_dir'], info_df=test_info, transform=test_transform, is_inference=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    predicts = np.zeros((len(test_loader.dataset), 500))

    # 모델 설정
    for result_path in config['result_paths']:
        exp = get_exp_json(result_path)
        model_selector = ModelSelector(model_type="timm", num_classes=config['num_classes'], model_name=exp['model_name'], pretrained=False)
        model = model_selector.get_model()

        # 베스트 모델 경로 설정
        model_path = get_best_model_path(result_path)
        print(f"Loading best model from {model_path}")

        # 저장된 모델 로드
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        # 추론 실행
        predict = inference(model, device, test_loader)
        predicts += predict


    # 결과 저장 디렉터리 확인 및 생성
    output_dir = os.path.dirname(config['output_path'])
    ensure_dir_exists(output_dir)

     # 예측 결과 저장
    predicts = predicts / len(config['result_paths'])
    target = np.argmax(predicts, axis = 1)
    test_info['target'] = target
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    
    softmax_info = pd.DataFrame(predicts, columns=[f'S{x}' for x in range(predicts.shape[1])])
    softmax_info = pd.concat((test_info, softmax_info), axis=1)

    test_info.to_csv(config['output_path'], index=False)
    softmax_info.to_csv(config['output_softmax_path'], index=False, float_format = "%.3e")

    print(f"Predictions saved to {config['output_path']}")