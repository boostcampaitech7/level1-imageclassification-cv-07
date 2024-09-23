#!/bin/bash

# .env 파일에서 환경 변수 로드
set -a
source .env
set +a

# W&B 로그인을 먼저 수행해야 함 (첫 실행 시)
wandb login "${WANDB_API_KEY}"
# config.json을 이용해 학습을 실행
python /data/ephemeral/home/Jeongseon/level1-imageclassification-cv-07/baseline_model/train.py --config /data/ephemeral/home/Jeongseon/level1-imageclassification-cv-07/baseline_model/config.json
python /data/ephemeral/home/Jeongseon/level1-imageclassification-cv-07/baseline_model/inference.py --config /data/ephemeral/home/Jeongseon/level1-imageclassification-cv-07/baseline_model/config.json

# chmod +x /data/ephemeral/home/Jeongseon/level1-imageclassification-cv-07/baseline_model/train.sh
# /data/ephemeral/home/Jeongseon/level1-imageclassification-cv-07/baseline_model/train.sh