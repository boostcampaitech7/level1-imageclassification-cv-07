import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torchvision import transforms
import cv2

def apply_erosion(image, **kwargs):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def apply_dilation(image, **kwargs):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

class AlbumentationsTransform:
    def __init__(self, is_train: bool = True):
        common_transforms = [
            A.Resize(224, 224),
            #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.Normalize(mean=[0.861, 0.861, 0.861], std=[0.253, 0.253, 0.253]),
            ToTensorV2() #자동 스케일 X, cf) ToTensor()는 자동 스케일링
        ]
        if is_train:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
<<<<<<< HEAD:baseline_model/datasets/transform.py
                    #A.Rotate(limit=15), #기하학적 변환(회전)
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.ElasticTransform(alpha=0.5, sigma=50, alpha_affine=20, p=0.5),   #기하학적 변환
                    A.Lambda(image=apply_erosion, p=0.3),  # Erosion 적용
                    A.Lambda(image=apply_dilation, p=0.3),  # Dilation 적용
                    A.OneOf([
                        A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.5),
                        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                    ], p=0.3),
                    #A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=16, min_width=16, p=1.0), # coarseDropout 추가 
=======
                    A.Rotate(limit=15),
>>>>>>> 36c41bc3cbddc0afad5d0ec6fa596c62a165a365:Model/0923_skf_analysis/datasets/transform.py
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
