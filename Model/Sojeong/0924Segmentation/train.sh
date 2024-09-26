#!/bin/bash

# W&B 로그인을 먼저 수행해야 함 (첫 실행 시)
wandb login (key 넣기)
# config.json을 이용해 학습을 실행
python /data/ephemeral/home/Sojeong/level1-imageclassification-cv-07/LV1/train.py --config /data/ephemeral/home/Sojeong/level1-imageclassification-cv-07/LV1/config.json
python /data/ephemeral/home/Sojeong/level1-imageclassification-cv-07/LV1/inference.py --config /data/ephemeral/home/Sojeong/level1-imageclassification-cv-07/LV1/config.json

# chmod +x /data/ephemeral/home/Sojeong/level1-imageclassification-cv-07/LV1/train.sh
# /data/ephemeral/home/Sojeong/level1-imageclassification-cv-07/LV1/train.sh