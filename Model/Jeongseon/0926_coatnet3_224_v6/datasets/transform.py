import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import RandAugment
import cv2
from PIL import Image


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
                    A.Rotate(limit=15), #기하학적 변환(회전)
                    A.RandAugment(num_ops=2, magnitude=5),
                    A.RandomBrightnessContrast(p=0.2),
                    #A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=16, min_width=16, p=1.0), # coarseDropout 추가 
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
                #transforms.Lambda(lambda image: Image.fromarray(np.uint8(image)) if isinstance(image, np.ndarray) else image), # numpy -> PIL 이미지로 변환
                #Image.fromarray(np.uint8(image)), # numpy -> PIL 이미지로 변환
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandAugment(num_ops=2, magnitude=5), # RandAugment 추가
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.861, 0.861, 0.861], std=[0.253, 0.253, 0.253])
                
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.861, 0.861, 0.861], std=[0.253, 0.253, 0.253])
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


