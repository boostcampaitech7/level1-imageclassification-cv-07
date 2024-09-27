import argparse
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
import wandb  # W&B 추가
from sklearn.model_selection import train_test_split
import os
import traceback  # 오류 스택 트레이스를 캡처하기 위한 모듈
import json
import utils.utils
from dotenv import load_dotenv
import subprocess
import csv


# 메인 학습 함수
def main(exp, config):
    try:
        # W&B 초기화
        wandb.init(project=config['wandb_project'], 
                   config=exp,
                   name=f"{exp['model_name']}_{exp['person_name']}_{exp['version']}",
                   entity='luckyvicky'
                   )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 데이터 로드
        train_info = pd.read_csv(config['data_info_file'])
        
        if exp['random_select_dataset']:
            # 데이터셋을 train과 valid로 나눔 
            train_df, val_df = train_test_split(train_info, test_size=exp['valid_split_ratio'], stratify=train_info['target'])
        else:
            train_index = pd.read_csv(exp['train_index_path'], header = None).squeeze()
            val_index = pd.read_csv(exp['valid_index_path'], header = None).squeeze()
            train_df = train_info.loc[train_index]
            val_df = train_info.loc[val_index]

        if exp['pseudo_label']:
            pseudo_info = pd.read_csv(exp["pseudo_label_path"])
            pseudo_info = pseudo_info.drop(columns = ['ID'])
            pseudo_info['pseudo_label'] = 1
            train_df['pseudo_label'] = 0
            train_df = pd.concat([train_df, pseudo_info]).reset_index(drop = True)

        # 변환 설정 (albumentations 사용)
        transform_selector = TransformSelector(transform_type="albumentations")
        train_transform = transform_selector.get_transform(is_train=True)
        val_transform = transform_selector.get_transform(is_train=False)

        # 데이터셋 및 데이터로더 생성 (train, valid)
        train_dataset = CustomDataset(exp, config, info_df=train_df, transform=train_transform)
        val_dataset = CustomDataset(exp, config, info_df=val_df, transform=val_transform)
        train_loader = DataLoader(train_dataset, batch_size=exp['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=exp['batch_size'], shuffle=False)

        # 모델 설정
        if exp['load_weight']:
            model_selector = ModelSelector(model_type="timm", num_classes=len(train_info['target'].unique()), model_name=exp['model_name'], pretrained=False)
            model = model_selector.get_model()
            model.load_state_dict(torch.load(exp['weight_path'], map_location=device))
        else:
            model_selector = ModelSelector(model_type="timm", num_classes=len(train_info['target'].unique()), model_name=exp['model_name'], pretrained=True)
            model = model_selector.get_model()

        model.to(device)

        # 옵티마이저 및 스케줄러
        if (exp['optimizer'] in ['SGD', 'sgd', 'Sgd']):
            optimizer = optim.SGD(model.parameters(), lr=exp['learning_rate'])
        elif(exp['optimizer'] in ['Adam', 'adam', 'ADAM']):
            optimizer = optim.Adam(model.parameters(), lr=exp['learning_rate'])

        scheduler = StepLR(optimizer, step_size = exp['StepLR_step_size'] * len(train_loader), gamma = exp['stepLR_gamma'])
        loss_fn = nn.CrossEntropyLoss(label_smoothing = exp['label_smoothing'])

        # Trainer 설정
        trainer = Trainer(model, device, train_loader, val_loader, optimizer, scheduler, loss_fn, exp['epochs'], exp['result_path'])

        # 학습 과정에서 W&B 로깅 추가
        for epoch in range(exp['epochs']):
            print(f"Epoch {epoch+1}/{exp['epochs']}")
            train_loss, train_acc = trainer.train_epoch()
            val_loss, val_acc = trainer.validate()

            log = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]['lr']
            }
            
            # W&B에 로그 기록
            wandb.log(log)

            # 텍스트 파일에 기록
            if (epoch == 0):
                f = open(exp['log_path'], "w")
                log_writer = csv.DictWriter(f, log.keys())
                log_writer.writeheader()

            log_writer.writerow(log)
            f.flush()

            # 모델 저장
            trainer.save_model(epoch, val_loss, exp['num_model_save'])

            # W&B 모델 가중치 업로드
            #wandb.save(os.path.join(config['result_path'], f"model_epoch_{epoch}.pt"))

        # 학습 완료 후 Slack DM 전송
        message = f"모델 학습이 완료되었습니다! Model: {exp['model_name']}, Epochs: {exp['epochs']}, Batch Size: {exp['batch_size']}"
        config['slack'].send_dm(message)
        # 로그 스트림 닫기
        f.close()

    except Exception as e:
        # 오류 메시지를 슬랙으로 전송
        error_message = f"Error occurred during training: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        config['slack'].send_dm(error_message)
        raise  # 예외를 다시 발생시켜 로그에 기록되도록 함
