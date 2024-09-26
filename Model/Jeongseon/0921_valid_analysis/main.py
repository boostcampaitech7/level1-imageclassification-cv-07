# %%
import json
import torch
from datasets.custom_dataset import CustomDataset
from datasets.transform import TransformSelector
from models.model_selector import ModelSelector
from utils.train_utils import Trainer
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt

# %%
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

def validate(model, device, val_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.08)
    total_batches = len(val_loader)
    
    progress_bar = tqdm(val_loader, desc="Validating", leave=False)
    
    with torch.no_grad():
        for batch_idx, (images, targets, indices) in enumerate(progress_bar):
            images, targets = images.to(device), targets.to(device)
            
            # 모델 예측
            outputs = model(images)
            outputs_softmax = torch.nn.functional.softmax(outputs)
            
            # 손실 계산
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            
            # 정확도 계산
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            progress_bar.set_postfix({'val_batch_loss': loss.item()})

            # 이미지 그리기
            top_fives, top_fives_indices = torch.topk(outputs_softmax, 5)

            images = images.cpu().numpy()
            targets = targets.cpu().numpy()
            indices = indices.cpu().numpy()
            outputs_softmax = outputs_softmax.cpu().numpy()
            top_fives = top_fives.cpu().numpy()
            top_fives_indices = top_fives_indices.cpu().numpy()                    
    
    # 검증의 평균 손실과 정확도 계산
    avg_loss = total_loss / total_batches
    accuracy = correct / total
    
    print(f"Validation Epoch Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return avg_loss, accuracy

# %%
config_path = r"/data/ephemeral/home/Dongjin/git/level1-imageclassification-cv-07/LV1/config_dj1.json"
with open(config_path, 'r') as f:
  config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_info = pd.read_csv(config['data_info_file'])

py_dir_path = r"/data/ephemeral/home/Dongjin/git/level1-imageclassification-cv-07/Model/0921_valid_analysis"
rel_train_index_path = r"datasets/train_index.csv"
rel_val_index_path = r"datasets/val_index.csv"

train_index_path = os.path.join(py_dir_path, rel_train_index_path)
val_index_path = os.path.join(py_dir_path, rel_val_index_path)

# train_index.csv와 val_index.csv를 이용하여 train_df와 val_df를 로드       
train_index = pd.read_csv(train_index_path, header = None).squeeze()
val_index = pd.read_csv(val_index_path, header = None).squeeze()

train_df = train_info.loc[train_index]
val_df = train_info.loc[val_index]

transform_selector = TransformSelector(transform_type="albumentations")
val_transform = transform_selector.get_transform(is_train=False)

val_dataset = CustomDataset(root_dir=config['train_data_dir'], info_df=val_df, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

# 모델 설정
model_selector = ModelSelector(model_type="timm", num_classes=config['num_classes'], model_name=config['model_name'], pretrained=False)
model = model_selector.get_model()

# 베스트 모델 경로 설정
model_path = get_best_model_path(config['result_path'])
print(f"Loading best model from {model_path}")

# 저장된 모델 로드
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

# %%
# 추론 실행
# validate(model, device, val_loader)

# %%
model.eval()
total_loss = 0.0
correct = 0
total = 0

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.08)
total_batches = len(val_loader)

progress_bar = tqdm(val_loader, desc="Validating", leave=False)

with torch.no_grad():
    for batch_idx, (images, targets, dataset_indices) in enumerate(progress_bar):
        images, targets = images.to(device), targets.to(device)
        
        # 모델 예측
        outputs = model(images)
        outputs_softmax = torch.nn.functional.softmax(outputs)
        
        # 손실 계산
        loss = loss_fn(outputs, targets)
        total_loss += loss.item()
        
        # 정확도 계산
        _, predicts = torch.max(outputs, 1)
        correct += (predicts == targets).sum().item()
        total += targets.size(0)
        
        progress_bar.set_postfix({'val_batch_loss': loss.item()})

        # 이미지 그리기
        top_fives, top_fives_indices = torch.topk(outputs_softmax, 5)

        images = images.cpu().numpy()
        targets = targets.cpu().numpy()
        dataset_indices = dataset_indices.cpu().numpy()
        outputs_softmax = outputs_softmax.cpu().numpy()
        top_fives = top_fives.cpu().numpy()
        top_fives_indices = top_fives_indices.cpu().numpy()
        predicts = predicts.cpu().numpy()

        break

# %%
for i in range(images.shape[0]):
    image = images[i]
    target = targets[i]
    dataset_index = dataset_indices[i] 
    output_softmax = outputs_softmax[i]
    top_five = top_fives[i]
    predict = predicts[i]


    




