# import os
# import json
# import torch
# import pandas as pd
# from datasets.custom_dataset import CustomDataset
# from datasets.transform import TransformSelector
# from models.model_selector import ModelSelector
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import argparse

# def get_best_model_path(directory):
#     files = [f for f in os.listdir(directory) if f.startswith('best_model') and f.endswith('.pt')]
#     if not files:
#         raise FileNotFoundError(f"No best model files found in directory: {directory}")
    
#     # 파일의 수정 시간을 기준으로 가장 최근에 저장된 파일 선택
#     best_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
#     return os.path.join(directory, best_file)



# from torchsummary import summary
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_selector = ModelSelector(model_type="timm", num_classes=500, model_name="coatnet_3_rw_224", pretrained=False)
# model = model_selector.get_model()

# # 베스트 모델 경로 설정
# model_path = get_best_model_path("/data/ephemeral/home/Seungcheol/level1-imageclassification-cv-07/LV1/results/coatnet_3_rw_224")
# print(f"Loading best model from {model_path}")

# # 저장된 모델 로드
# model.load_state_dict(torch.load(model_path, map_location=device))
# #summary(model, (3,224,224))
# num_params = sum([p.numel() for p in model.parameters()])
# trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
# print(model)
# print(f"{num_params = :,} | {trainable_params = :,}")