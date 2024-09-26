import torch
from torch.utils.data import Dataset

class SegmentationThenClassificationDataset(Dataset):
    def __init__(self, original_dataset, segmentation_model, device):
        """
        세그멘테이션 후 배경을 제거한 이미지를 반환하는 데이터셋 클래스
        :param original_dataset: 원본 CustomDataset 객체 (이미지와 라벨을 제공)
        :param segmentation_model: 사전 학습된 세그멘테이션 모델
        :param device: 사용할 장치 (CPU 또는 GPU)
        """
        self.original_dataset = original_dataset
        self.segmentation_model = segmentation_model
        self.device = device

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        # 원본 데이터셋에서 이미지와 라벨을 가져옴
        image, label = self.original_dataset[index]

        # 이미지를 세그멘테이션 모델에 입력
        image = image.to(self.device).unsqueeze(0)  # 배치 차원 추가
        with torch.no_grad():
            # 세그멘테이션 모델에서 출력을 얻음
            seg_output = self.segmentation_model(image)['out']
            seg_output = torch.sigmoid(seg_output)  # 시그모이드 활성화 함수 적용

        # 배경을 제거하기 위해 이진 마스크 생성 (배경 클래스: 0, 객체 클래스: 1)
        # 보통 첫 번째 채널이 배경을 나타냄, 이를 기반으로 마스크 생성
        binary_mask = seg_output.argmax(dim=1) != 0  # 배경(클래스 0)이 아닌 부분만 True로 설정
        binary_mask = binary_mask.squeeze(0).unsqueeze(0)  # 배치 차원을 제거 후 채널 추가

        # 원본 이미지에 마스크를 적용하여 배경 제거
        image = image.squeeze(0)  # 배치 차원 제거
        masked_image = image * binary_mask.float()  # 배경이 아닌 부분만 남김

        return masked_image, label
