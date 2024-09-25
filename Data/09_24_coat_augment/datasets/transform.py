import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class AlbumentationsTransform1:
    def __init__(self, is_train: bool = True):
        common_transforms = [
            A.Resize(256, 256),
            A.Normalize(mean=[0.865, 0.865, 0.865], std=[0.26, 0.26, 0.26]),
            ToTensorV2()
        ]
        if is_train:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15),
                    A.RandomBrightnessContrast(p=0.2),
                    A.RandomRotate90(),                # Randomly rotate by 90 degrees
                    A.Rotate(limit=15),                # Custom rotation
                    A.Affine(rotate=15,                # Custom affine transformation
                            translate_percent={"x": 0.1, "y": 0.1}),
                    A.HorizontalFlip(p=0.5),          # Custom horizontal flip
                    A.VerticalFlip(p=0.5),            # Custom vertical flip
                    A.RandomErasing(p=0.5,             # Random erase
                                    scale=(0.02, 0.2)),
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
    
    
class AlbumentationsTransform2:
    def __init__(self, is_train: bool = True):
        common_transforms = [
            A.Resize(512, 512),
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
            # Define a list of transformations, starting with AutoAugment
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),

                transforms.RandomRotation(15),         # Custom rotation
                transforms.RandomAffine(degrees=15,    # Custom affine transformation
                                        translate=(0.1, 0.1)),
                transforms.RandomHorizontalFlip(),     # Custom horizontal flip
                transforms.RandomVerticalFlip(),       # Custom vertical flip
                transforms.RandomErasing(scale=(0.02, 0.2)),  # Random erase

                transforms.Normalize(mean=[0.865, 0.865, 0.865], std=[0.26, 0.26, 0.26])
            ])
        else:
            # Simpler transform for test set
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.865, 0.865, 0.865], std=[0.26, 0.26, 0.26])
            ])

    def __call__(self, image):
        pil_image = Image.fromarray(image)
        return self.transform(pil_image)

class TransformSelector:
    def __init__(self, transform_type: str):
        if transform_type == "albumentations1":
            self.transform_type = "albumentations1"
        elif transform_type == "albumentations2":
            self.transform_type = "albumentations2"
        elif transform_type == "torchvision":
            self.transform_type = "torchvision"
        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self, is_train: bool):
        if self.transform_type == "albumentations1":
            return AlbumentationsTransform1(is_train=is_train)
        elif self.transform_type == "albumentations2":
            return AlbumentationsTransform2(is_train=is_train)
        elif self.transform_type == "torchvision":
            return TorchvisionTransform(is_train=is_train)


