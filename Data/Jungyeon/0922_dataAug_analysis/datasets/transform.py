import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

 # 데이터 수 증강
# class BezierPivotDeformation:
#     def __init__(self, control_points, num_samples=100):
#         self.control_points = control_points
#         self.num_samples = num_samples

#     def bezier_curve(self, t):
#         n = len(self.control_points) - 1
#         return sum(
#             (np.math.comb(n, i) * (1 - t) ** (n - i) * t ** i * np.array(self.control_points[i]))
#             for i in range(n + 1)
#         )

#     def apply(self, image):
#         # 베지어 곡선에 따른 변형 생성
#         t_values = np.linspace(0, 1, self.num_samples)
#         bezier_points = np.array([self.bezier_curve(t) for t in t_values], dtype=np.int32)

#         # 이미지 변형
#         mask = np.zeros_like(image)
#         cv2.fillConvexPoly(mask, bezier_points, (1, 1, 1))  # 변형할 영역 생성
#         deformed_image = image * mask  # 이미지에 적용
        
#         return deformed_image
    
# class MeanStrokeReconstruction:
#     def __init__(self, num_sketches=5):
#         self.num_sketches = num_sketches

#     def apply(self, image):
#         # 여러 번의 스케치 형태로 변환하여 평균을 계산
#         sketches = []
#         for _ in range(self.num_sketches):
#             # 스케치 효과 적용 (여기서는 간단히 흐림 처리 사용)
#             sketch = cv2.GaussianBlur(image, (5, 5), 0)
#             sketches.append(sketch)
#         # 평균 계산
#         return np.mean(sketches, axis=0).astype(np.uint8)

class AlbumentationsTransform:
    def __init__(self, is_train: bool = True):
        common_transforms = [
            A.Resize(224, 224),
            A.Normalize(mean=[0.865, 0.865, 0.865], std=[0.26, 0.26, 0.26]),
            ToTensorV2()
        ]
        # self.bpd = BezierPivotDeformation([(100, 100), (150, 50), (200, 100)])
        # self.msr = MeanStrokeReconstruction(num_sketches=5)


        
        if is_train:
            self.transform = A.Compose(
                [
                    # 기본 변환
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15),
                    A.ToGray(p=0.5),  # Grayscale 변환 추가
                    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=255, p=0.5),
                    A.Compose([A.Morphological(p=1, scale=(2, 3), operation='erosion')], p=1), # erosion
                    A.ElasticTransform(alpha=2.0, sigma=0.1, p=0.5)
                ] + common_transforms
            )
        else:
            self.transform = A.Compose(common_transforms)
            

    def __call__(self, image):
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array.")
        transformed = self.transform(image=image)
        return transformed['image']


class TorchvisionTransform:
    def __init__(self, is_train: bool = True):
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __call__(self, image):
        return self.transform(image)

class TransformSelector:
    def __init__(self, transform_type: str):
        if transform_type == "albumentations":
            self.transform_type = "albumentations"
        elif transform_type == "torchvision":
            self.transform_type = "torchvision"
        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self, is_train: bool):
        if self.transform_type == "albumentations":
            return AlbumentationsTransform(is_train=is_train)
        elif self.transform_type == "torchvision":
            return TorchvisionTransform(is_train=is_train)


