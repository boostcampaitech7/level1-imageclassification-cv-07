import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torchvision import transforms

class AlbumentationsTransform:
    def __init__(self, is_train: bool = True):
        common_transforms = [
            A.Resize(224, 224),
            A.Normalize(mean=[0.865, 0.865, 0.865], std=[0.26, 0.26, 0.26]),
            ToTensorV2()
        ]
        if is_train:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15),
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


