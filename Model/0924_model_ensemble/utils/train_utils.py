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
        
    def save_model(self, epoch, loss, num_save):
        # 결과 저장 경로 생성
        os.makedirs(self.result_path, exist_ok=True)

        # 현재 에폭의 모델 저장
        current_model_path = os.path.join(self.result_path, f'model_epoch_{epoch}_loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path) #state_dict(): 모델의 학습 가능한 가중치를 포함한 모델의 상태 사전 반환

        # 저장된 모델을 리스트에 추가 (손실 값, 에폭 번호, 모델 경로)
        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort(key=lambda x: x[0])  # 손실 값 기준으로 정렬 (낮은 순으로), key: 정렬 기준, x[0]: 손실 값

        # 모델이 3개를 넘으면 가장 높은 손실을 가진 모델 삭제
        if len(self.best_models) > num_save:
            _, _, path_to_remove = self.best_models.pop(-1) #
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
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False) #tqdm: 진행률 표시
        
        for batch_idx, (images, targets) in enumerate(progress_bar): #enumerate: 인덱스와 값을 반환(enumerate), batch_idx: 인덱스, images, targets: 값
            images, targets = images.to(self.device), targets.to(self.device)
            
            # 옵티마이저 초기화
            self.optimizer.zero_grad() 

            # 모델 예측
            outputs = self.model(images)
            
            # 손실 계산
            loss = self.loss_fn(outputs, targets)
            
            # 역전파 및 옵티마이저 스텝
            loss.backward()
            self.optimizer.step() #가중치 업데이트
            self.scheduler.step() #학습률 업데이트 (학습률 스케쥴러 한 단계 진행)
            
            # 배치 손실 및 정확도 계산
            total_loss += loss.item()       #item(): 텐서의 값 -> 스칼라 값으로 반환 (텐서 GPU -> CPU로 이동)
            _, predicted = torch.max(outputs, 1)  #torch.max : 최대값과 그 인덱스 반환 (1: 열 기준 최대값), 인덱스만 필요하므로 _로 최대값은 무시
            correct += (predicted == targets).sum().item()
            total += targets.size(0) #targets.size(0): 배치 크기
            
            progress_bar.set_postfix({'batch_loss': loss.item()})
            
        
        # 평균 손실과 정확도 계산
        avg_loss = total_loss / total_batches 
        accuracy = correct / total
        
        print(f"Training Epoch Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return avg_loss, accuracy

    # 검증 함수 (validate)
    def validate(self):
        self.model.eval() #모델을 평가 모드로 설정 (드롭아웃, 배치 정규화 등을 평가 모드로 설정)
        total_loss = 0.0
        correct = 0 #정답 수
        total = 0 #전체 데이터 수
        total_batches = len(self.val_loader) #검증 데이터셋 배치 수
        
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        with torch.no_grad(): #기울기 계산 비활성화
            for batch_idx, (images, targets) in enumerate(progress_bar):
                images, targets = images.to(self.device), targets.to(self.device)
                
                # 모델 예측
                outputs = self.model(images)
                
                # 손실 계산
                loss = self.loss_fn(outputs, targets) #배치의 손실 계산
                total_loss += loss.item() #누적 손실에 더함
                
                # 정확도 계산
                _, predicted = torch.max(outputs, 1) #예측 클래스 결정, output.shape= (batch_size, num_classes)에서 가장 큰 값과 그 인덱스 반환
                correct += (predicted == targets).sum().item() #정답 수 누적
                total += targets.size(0) #전체 데이터 수 누적
                
                progress_bar.set_postfix({'val_batch_loss': loss.item()})
                
        
        # 검증의 평균 손실과 정확도 계산
        avg_loss = total_loss / total_batches #전체 손실의 평균
        accuracy = correct / total #옳은 예측 수 / 전체 데이터 수
        
        print(f"Validation Epoch Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return avg_loss, accuracy
