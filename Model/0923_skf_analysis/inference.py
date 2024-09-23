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

# 추론(inference) 함수 + 앙상블 예측
def ensemble_predict(models, device, test_loader):
    predictions = np.zeros((len(dataloader.dataset), 500)) # 예측값을 저장할 배열을 초기화

    for model in models:
        model.eval()
        with torch.no_grad():
            fold_predictions = [] # 각 모델의 예측값을 저장할 리스트를 초기화
            for X, _ in dataloader:
                X = X.to(device)
                pred = model(X)
                fold_predictions.append(pred.cpu().numpy())
            fold_predictions = np.concatenate(fold_predictions, axis=0) # 예측값을 합산하여 앙상블
            predictions += fold_predictions
    return predictions # 최종 예측값 반환

# 'best_model' 파일을 찾는 함수 (가장 최근에 저장된 파일 선택)
def get_model_paths(directory):
    files = [f for f in os.listdir(directory) if f.startswith('model_epoch') and f.endswith('.pt')]
    if not files:
        raise FileNotFoundError(f"No best model files found in directory: {directory}")
    
    # 파일의 수정 시간을 기준으로 가장 최근에 저장된 파일 선택
    # best_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    paths = [os.path.join(directory, f) for f in files]
    return paths
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

    # 테스트 데이터셋 및 데이터로더 생성
    test_dataset = CustomDataset(root_dir=config['test_data_dir'], info_df=test_info, transform=test_transform, is_inference=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # 모델 설정
    model_selector = ModelSelector(model_type="timm", num_classes=config['num_classes'], model_name=config['model_name'], pretrained=False)
    model = model_selector.get_model()

    # # 베스트 모델 경로 설정
    model_paths = get_model_paths(config['result_path'])
    print(f"Loading models from {model_paths}")

    # # 저장된 모델 로드
    models = []
    for model_path in model_paths:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        models.append(model)

    # 추론 실행
    test_predictions = ensemble_predict(models, test_dataloader, device)

    # 결과 저장 디렉터리 확인 및 생성
    output_dir = os.path.dirname(config['output_path'])
    ensure_dir_exists(output_dir)

    # 예측 결과 저장
    test_info['target'] = predictions
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
