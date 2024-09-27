import torch 
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights  # DeepLabV3 모델 사용
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import numpy as np
import os 
from PIL import Image

class SegmentationModel:
    def __init__(self, device):
        """
        세그멘테이션 모델 초기화
        :param device: 사용할 장치 (CPU 또는 GPU)
        """
        # 사전 학습된 DeepLabV3 모델 로드
        #weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 # COCO 데이터셋으로 학습된 가중치 사용
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.model = deeplabv3_resnet50(pretrained=True).to(device)
        self.model.eval()  # 평가 모드로 설정
        self.device = device

    def segment_image(self, image):
        """
        이미지를 세그멘테이션 모델에 입력하고 결과를 반환
        :param image: 입력 이미지 (Tensor)
        :return: 세그멘테이션 결과 (Tensor)
        """
        image = image.to(self.device).unsqueeze(0)  # 배치 차원 추가
        with torch.no_grad():
            # 세그멘테이션 모델 적용
            seg_output = self.model(image)['out']
            seg_output = torch.sigmoid(seg_output)  # 활성화 함수 적용
        return seg_output.squeeze(0)  # 배치 차원 제거

    def save_segmentation(self, original_image, seg_output, save_dir="output", file_prefix="segmented_image"):
        """
        세그멘테이션 결과 저장
        :param original_image: 원본 이미지 (Tensor)
        :param seg_output: 세그멘테이션 모델의 출력 (Tensor)
        :param save_dir: 이미지를 저장할 디렉토리
        :param file_prefix: 저장 파일의 접두사
        """
        # 저장할 디렉토리 생성 (존재하지 않으면 생성)
        os.makedirs(save_dir, exist_ok=True)

        # 원본 이미지를 numpy 배열로 변환
        original_image_np = F.to_pil_image(original_image.cpu()).convert("RGB")
        
        # 세그멘테이션 마스크에서 가장 높은 값을 가진 인덱스(클래스)를 선택
        mask = seg_output.argmax(dim=0).cpu().numpy()
        print(mask)

        # 배경은 0으로, 객체는 1로 설정하여 이진 마스크 생성
        binary_mask = mask != 0

        # 마스크를 이용해 원본 이미지의 배경을 제거한 masked_image 생성
        original_image_np = np.array(original_image_np)
        masked_image = original_image_np * np.expand_dims(binary_mask, axis=-1)
        print(masked_image)
        
        # PIL 객체로 변환하여 저장
        original_image_pil = Image.fromarray(original_image_np)
        mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))  # 마스크를 0~255로 변환하여 저장
        masked_image_pil = Image.fromarray(masked_image.astype(np.uint8))

        # 파일 저장
        original_image_pil.save(os.path.join(save_dir, f"{file_prefix}_original.png"))
        mask_pil.save(os.path.join(save_dir, f"{file_prefix}_mask.png"))
        masked_image_pil.save(os.path.join(save_dir, f"{file_prefix}_masked.png"))

        print(f"Segmentation results saved to {save_dir}")