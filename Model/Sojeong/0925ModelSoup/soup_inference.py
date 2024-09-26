import os
import json
import torch
import pandas as pd
from datasets.custom_dataset import CustomDataset
from datasets.transform import TransformSelector
from models.model_selector import ModelSelector
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np 
import argparse

# 모델 수프 함수 (Uniform Soup)
def uniform_soup(model_paths, test_loader, model_fun, device):
    soups = []
    model = model_fun().to(device)
    
    # 각 모델의 가중치를 수프 리스트에 추가
    for i, path in enumerate(model_paths):
        print(f"Loading model {i} from: {path}")  # 모델 파일 경로 출력
        model.load_state_dict(torch.load(path, map_location=device))
        soup = [param.detach().cpu().numpy() for param in model.parameters()]
        
        # 가중치 크기 출력 및 총합 계산
        total_params_sum = sum([np.prod(param.shape) for param in soup])
        print(f"Model {i} weights shape sum: {total_params_sum}")
        
        soups.append(soup)
    
    # 평균화 가능한 가중치만 처리
    mean_soup = []
    for idx, params in enumerate(zip(*soups)):
        param_shapes = [param.shape for param in params]
        
        # 모든 모델의 해당 레이어가 동일한 형태를 가질 경우에만 평균화
        if all(shape == param_shapes[0] for shape in param_shapes):
            avg_param = np.mean(np.array(params), axis=0)
            mean_soup.append(avg_param)
        else:
            print(f"Shape mismatch in layer {idx}:")
            for model_idx, shape in enumerate(param_shapes):
                print(f"  Model {model_idx}: {shape}")
    
    # 평균화된 가중치를 모델에 적용
    for param, avg_weight in zip(model.parameters(), mean_soup):
        param.data = torch.tensor(avg_weight).to(device)
    
    return model


# 추론 함수
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

# 'best_model' 파일들을 찾는 함수 (디렉토리 내 모델 파일 리스트 반환)
def get_model_paths(directory):
    files = [f for f in os.listdir(directory) if f.startswith('best_model') and f.endswith('.pt')]
    if not files:
        raise FileNotFoundError(f"No best model files found in directory: {directory}")
    
    model_paths = [os.path.join(directory, f) for f in files]
    return model_paths

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
    model_fun = model_selector.get_model

    # 모든 'best_model' 경로 가져오기
    model_paths = get_model_paths(config['result_path'])
    print(f"Found {len(model_paths)} model files for soup")

    # 모델 수프 생성 (Uniform Soup)
    soup_model = uniform_soup(model_paths, test_loader, model_fun, device)

    # 추론 실행
    predictions = inference(soup_model, device, test_loader)

    # 결과 저장 디렉터리 확인 및 생성
    output_dir = os.path.dirname(config['output_path'])
    ensure_dir_exists(output_dir)

    # 예측 결과 저장
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv(config['output_path'], index=False)
    print(f"Predictions saved to {config['output_path']}")

    # 모델 저장 디렉터리 생성
    model_save_dir = config.get('model_save_dir', './saved_models/')
    ensure_dir_exists(model_save_dir)

    # 모델 가중치 저장
    model_save_path = os.path.join(model_save_dir, 'soup_model.pt')
    torch.save(soup_model.state_dict(), model_save_path)
    print(f"Soup model saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference using model soup")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config)

# python /data/ephemeral/home/Sojeong/level1-imageclassification-cv-07/Model/0925ModelSoup/soup_inference.py --config /data/ephemeral/home/Sojeong/level1-imageclassification-cv-07/Model/0925ModelSoup/config.json