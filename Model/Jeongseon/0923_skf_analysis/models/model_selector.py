import timm
import torch.nn as nn

class TimmModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super(TimmModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

class ModelSelector:
    def __init__(self, model_type: str, num_classes: int, model_name: str, pretrained: bool = True):
        if model_type == 'timm':
            self.model = TimmModel(model_name, num_classes, pretrained)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def get_model(self):
        return self.model