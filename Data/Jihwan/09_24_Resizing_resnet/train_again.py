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
import pandas as pd
import requests
import wandb  # W&B 추가
from sklearn.model_selection import train_test_split
import os
import traceback  # 오류 스택 트레이스를 캡처하기 위한 모듈
import timm

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
                   name=f"{config['model_name']}_{config['person_name']}_{config['version']}",
                   entity='luckyvicky'
                   )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 데이터 로드
        train_info = pd.read_csv(config['data_info_file'])
        
        # # 데이터셋을 train과 valid로 나눔 
        
        train_df, val_df = train_test_split(train_info, test_size=0.1, stratify=train_info['target'], random_state=0)

        """
        # train_index.csv와 val_index.csv 경로
        py_dir_path = os.path.dirname(os.path.abspath(__file__))
        rel_train_index_path = os.path.normpath("datasets/train_index.csv")
        rel_val_index_path = os.path.normpath("datasets/val_index.csv")

        train_index_path = os.path.join(py_dir_path, rel_train_index_path)
        val_index_path = os.path.join(py_dir_path, rel_val_index_path)

        # # train_index.csv와 val_index.csv 저장
        # train_df.index.to_series().to_csv(train_index_path, index = False, header = False)
        # val_df.index.to_series().to_csv(val_index_path, index = False, header = False)

        # train_index.csv와 val_index.csv를 이용하여 train_df와 val_df를 로드       
        train_index = pd.read_csv(train_index_path, header = None).squeeze()
        val_index = pd.read_csv(val_index_path, header = None).squeeze()

        train_df = train_info.loc[train_index]
        val_df = train_info.loc[val_index]

        """
        # 변환 설정 (albumentations 사용)
        transform_selector = TransformSelector(transform_type="albumentations3")
        train_transform = transform_selector.get_transform(is_train=True)
        val_transform = transform_selector.get_transform(is_train=False)

        # 데이터셋 및 데이터로더 생성 (train, valid)
        train_dataset = CustomDataset(root_dir=config['train_data_dir'], info_df=train_df, transform=train_transform)
        val_dataset = CustomDataset(root_dir=config['train_data_dir'], info_df=val_df, transform=val_transform)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        # 모델 설정
        #model_selector = ModelSelector(model_type="timm", num_classes=len(train_info['target'].unique()), model_name=config['model_name'], pretrained=True)
        #model = model_selector.get_model()
        
        model = timm.create_model("wide_resnet50_2", num_classes=500, pretrained=True)
        model_path = '/data/ephemeral/home/Jihwan/level1-imageclassification-cv-07/Data/09_24_Resizing_resnet/results/best_model_1.9271.pt'

        # Load the state dictionary from the file
        state_dict = torch.load(model_path)
        # new_state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith('model.')}

        # Remove the 'model.' prefix from the keys

        # Load the modified state dictionary into the model
        model.load_state_dict(state_dict)
        
        model.to(device)

        # 옵티마이저 및 스케줄러
        optimizer = optim.SGD(model.parameters(), lr=0.009)
        scheduler = StepLR(optimizer, step_size=2 * len(train_loader), gamma=0.9)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

        # Trainer 설정
        trainer = Trainer(model, device, train_loader, val_loader, optimizer, scheduler, loss_fn, config['epochs'], config['result_path'])

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
            trainer.save_model(epoch, val_loss)

            # W&B 모델 가중치 업로드
            #wandb.save(os.path.join(config['result_path'], f"model_epoch_{epoch}.pt"))
            
            
            """
            if epoch == 9:
                
                # 변환 설정 (albumentations 사용)
                transform_selector = TransformSelector(transform_type="albumentations2")
                train_transform = transform_selector.get_transform(is_train=True)
                val_transform = transform_selector.get_transform(is_train=False)

                # 데이터셋 및 데이터로더 생성 (train, valid)
                train_dataset = CustomDataset(root_dir=config['train_data_dir'], info_df=train_df, transform=train_transform)
                val_dataset = CustomDataset(root_dir=config['train_data_dir'], info_df=val_df, transform=val_transform)
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
                
                optimizer = optim.SGD(model.parameters(), lr=0.009)
                scheduler = StepLR(optimizer, step_size=2 * len(train_loader), gamma=0.9)
                loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
                trainer = Trainer(model, device, train_loader, val_loader, optimizer, scheduler, loss_fn, config['epochs'], config['result_path'])
            """

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