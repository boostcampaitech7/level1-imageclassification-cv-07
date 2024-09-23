import torch
import os
from tqdm import tqdm

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

            
    # 훈련 함수 (train_epoch)
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images, targets = images.to(self.device), targets.to(self.device)
            
            # 옵티마이저 초기화
            self.optimizer.zero_grad()
            
            # 모델 예측
            outputs = self.model(images)
            
            # 손실 계산
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

    def train(self):
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
