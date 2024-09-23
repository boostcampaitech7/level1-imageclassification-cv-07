import torch
import os
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, model, device, train_loader, val_loader, optimizer, scheduler, loss_fn, epochs, result_path):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.result_path = result_path
        self.best_models = []  # 저장된 모델 목록 (손실 값, 에폭, 경로)
        self.lowest_loss = float('inf')  # 가장 낮은 검증 손실을 기록하기 위한 초기값 설정
        
    def save_model(self, epoch, loss):
        # 결과 저장 경로 생성
        os.makedirs(self.result_path, exist_ok=True)

        # 현재 에폭의 모델 저장
        current_model_path = os.path.join(self.result_path, f'model_epoch_{epoch}_loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)

        # 저장된 모델을 리스트에 추가 (손실 값, 에폭 번호, 모델 경로)
        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort(key=lambda x: x[0])  # 손실 값 기준으로 정렬 (낮은 순으로)

        # 모델이 3개를 넘으면 가장 높은 손실을 가진 모델 삭제
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(-1)
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)
                print(f"Model at {path_to_remove} removed due to exceeding the model limit.")

        # 가장 낮은 손실 모델을 별도로 저장 (기존 best_model이 있으면 삭제)
        best_model_path = os.path.join(self.result_path, f'best_model_{loss:.4f}.pt')

        if loss < self.lowest_loss:
            # 기존 best_model 파일 삭제
            previous_best_model_path = os.path.join(self.result_path, f'best_model_{self.lowest_loss:.4f}.pt')
            if os.path.exists(previous_best_model_path):
                os.remove(previous_best_model_path)
                print(f"Previous best model {previous_best_model_path} removed.")

            # 새로운 best_model 저장
            self.lowest_loss = loss
            torch.save(self.model.state_dict(), best_model_path)
            print(f"New best model saved at {best_model_path} with loss = {loss:.4f}")

    # MixUp 구현
    def mixup_data(x, y, alpha=1.0):
        '''입력과 라벨을 섞고 lambda 값을 반환'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    # 랜덤 영역을 잘라내기 위한 함수
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
    
    # CutMix 구현
    def cutmix_data(self, x, y, alpha=1.0):
        '''입력과 라벨을 섞고 lambda 값을 반환'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam

    def freeze_model_layers(self, model):
        for name, param in model.named_parameters():
            # stem, stage 동결
            if 'stem' in name or 'stages.0' or 'stages.1' in name:
                param.requires_grad = False # early layer freeze
            else:
                param.requires_grad = True # unfrozen

        # # freeze된 파라미터 확인
        # for name, param in model.named_parameters():
        #     if not param.requires_grad:
        #         print(f"Frozen layer: {name}")
        #     else:
        #         print(f"Unfrozen layer: {name}")
        # print("here")

    def classifier_unfreeze_layers(self, model):
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                # freeze된 파라미터 확인
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"Frozen layer: {name}")
            else:
                print(f"Unfrozen layer: {name}")
        print("here")

            
    # 훈련 함수 (train_epoch)
    def train_epoch(self, use_cutmix="False", use_mixup="False", alpha=1.0):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images, targets = images.to(self.device), targets.to(self.device)
            
            # CutMix 또는 MixUp 적용
            if use_cutmix == 'True':
                images, targets_a, targets_b, lam = self.cutmix_data(images, targets, alpha)
            elif use_mixup == 'True':
                images, targets_a, targets_b, lam = self.mixup_data(images, targets, alpha)

            # 옵티마이저 초기화
            self.optimizer.zero_grad()
            
            # 모델 예측
            outputs = self.model(images)
            
            # 손실 계산
            if use_cutmix == 'True' or use_mixup == 'True':
                loss = lam * self.loss_fn(outputs, targets_a) + (1 - lam) * self.loss_fn(outputs, targets_b)
            else:
                loss = self.loss_fn(outputs, targets)
            
            
            # 역전파 및 옵티마이저 스텝
            loss.backward()
            self.optimizer.step()
            
            # 배치 손실 및 정확도 계산
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            progress_bar.set_postfix({'batch_loss': loss.item()})
        
        # 평균 손실과 정확도 계산
        avg_loss = total_loss / total_batches
        accuracy = correct / total
        
        print(f"Training Epoch Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return avg_loss, accuracy

    # 검증 함수 (validate)
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(self.val_loader)
        
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(progress_bar):
                images, targets = images.to(self.device), targets.to(self.device)
                
                # 모델 예측
                outputs = self.model(images)
                
                # 손실 계산
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                
                # 정확도 계산
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
                
                progress_bar.set_postfix({'val_batch_loss': loss.item()})
        
        # 검증의 평균 손실과 정확도 계산
        avg_loss = total_loss / total_batches
        accuracy = correct / total
        
        print(f"Validation Epoch Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return avg_loss, accuracy

    # def train(self):
    #     for epoch in range(self.epochs):
    #         print('epoch')
    #         # epoch 5까지는 freeze layer
    #         if epoch < 5:
    #             self.freeze_model_layers(self.model)
                
    #         train_loss, train_acc = self.train_epoch()
    #         val_loss, val_acc = self.validate()
    #         self.scheduler.step()
    #         print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
    #               f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
