import timm
import torch.nn as nn

class TimmModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super(TimmModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        #모델의 모든 모듈 탐색
        # count2=0
        # for name, module in self.model.named_modules():
        #     if isinstance(module, nn.Dropout):
        #         count2+=1
        #         if count2==49:
        #             module.p = 0.5  # 원하는 확률로 변경 (예: 0.5)

        # # 모델의 모든 모듈 탐색
        # for name, module in self.model.named_modules():
        #     if isinstance(module, nn.Dropout):
        #         #print(f"Current Dropout in {name}: {module.p}")  # 현재 Dropout 확률 출력
        #         module.p = 0.2  # 원하는 확률로 변경 (예: 0.5)

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
