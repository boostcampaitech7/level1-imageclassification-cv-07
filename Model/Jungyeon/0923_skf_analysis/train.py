import argparse
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
import numpy as np
import pandas as pd
import requests
import wandb  # W&B 추가
from sklearn.model_selection import train_test_split
import os
import traceback  # 오류 스택 트레이스를 캡처하기 위한 모듈
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from torchvision.transforms import v2



# Slack 알림 함수 (DM용)
def send_slack_dm(token: str, user_id: str, message: str):
    url = 'https://slack.com/api/conversations.open'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {'users': [user_id]}
    
    # 채널 열기
    open_response = requests.post(url, headers=headers, data=json.dumps(data))
    open_data = open_response.json()
    if not open_data.get('ok'):
        raise Exception(f"Slack API Error: {open_data.get('error')}")
    
    channel_id = open_data['channel']['id']
    
    # 메시지 보내기
    message_url = 'https://slack.com/api/chat.postMessage'
    message_data = {
        'channel': channel_id,
        'text': message
    }
    message_response = requests.post(message_url, headers=headers, data=json.dumps(message_data))
    message_data = message_response.json()
    if not message_data.get('ok'):
        raise Exception(f"Slack API Error in chat.postMessage: {message_data.get('error')}")

# 메인 학습 함수
def main(config):
    try:
        # W&B 초기화
        wandb.init(project=config['wandb_project'], 
                   config=config,
                   entity="luckyvicky",
                   name=f"{config['model_name']}_{config['person_name']}_{config['version']}"
                   )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 데이터 로드
        train_info = pd.read_csv(config['data_info_file'])
        
        # StratifiedKFold 설정
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        labels = [ ]
        for data in train_info['target']:
            labels.append(data)
            
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            print(f'Fold {fold+1}')

            # train_idx와 val_idx에 따른 학습 및 검증 데이터 분리
            print(f'train_idx : {train_idx}, val_idx : {val_idx}')
            train_subset = train_info.iloc[train_idx].reset_index(drop=True)
            val_subset = train_info.iloc[val_idx].reset_index(drop=True)
            
            # 변환 설정 (albumentations 사용)
            transform_selector = TransformSelector(transform_type="albumentations")
            train_transform = transform_selector.get_transform(is_train=True)
            val_transform = transform_selector.get_transform(is_train=False)

            # CustomDataset 생성
            train_dataset = CustomDataset(root_dir=config['train_data_dir'], info_df=train_subset, transform=train_transform)
            val_dataset = CustomDataset(root_dir=config['train_data_dir'], info_df=val_subset, transform=val_transform)

            # DataLoader 생성
            train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
            
            cutmix = v2.CutMix(num_classes=500)
            mixup = v2.MixUp(num_classes=500)
            cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
            
            for images, labels in train_dataloader:
                images, labels = cutmix_or_mixup(images, labels)

            # 모델 생성 및 학습
            model_selector = ModelSelector(model_type="timm", num_classes=len(train_info['target'].unique()), model_name=config['model_name'], pretrained=True)
            model = model_selector.get_model()
            model.to(device)

            # 옵티마이저 + scheduler
            optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
            scheduler = StepLR(optimizer, step_size=2 * len(train_dataloader), gamma=0.5)
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

            # Trainer 설정
            trainer = Trainer(model, device, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, config['epochs'], config['result_path'])
            
            # 학습 과정에서 W&B 로깅 추가
            for epoch in range(config['epochs']):
                print(f"Epoch {epoch+1}/{config['epochs']}")
                train_loss, train_acc = trainer.train_epoch()
                val_loss, val_acc = trainer.validate()

                # W&B에 로그 기록
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                
                # 모델 저장
                trainer.save_model(fold, epoch, val_loss)
                


        # 학습 완료 후 Slack DM 전송
        slack_token = config['slack_token']
        slack_user_id = config['slack_user_id']
        message = f"모델 학습이 완료되었습니다! Model: {config['model_name']}, Epochs: {config['epochs']}, Batch Size: {config['batch_size']}"
        send_slack_dm(slack_token, slack_user_id, message)

    except Exception as e:
        # 오류 메시지를 슬랙으로 전송
        error_message = f"Error occurred during training: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        send_slack_dm(config['slack_token'], config['slack_user_id'], error_message)
        raise  # 예외를 다시 발생시켜 로그에 기록되도록 함

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model using config file")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config)