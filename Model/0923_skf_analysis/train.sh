#!/bin/bash

# W&B 로그인을 먼저 수행해야 함 (첫 실행 시)
wandb login 3a6ad2d8b554254e6a837236f48ea68235f4f427
# config.json을 이용해 학습을 실행
python /data/ephemeral/home/Jungyeon/level1-imageclassification-cv-07/Model/0923_skf_analysis/train.py --config /data/ephemeral/home/Jungyeon/level1-imageclassification-cv-07/Model/0923_skf_analysis/config.json
python /data/ephemeral/home/Jungyeon/level1-imageclassification-cv-07/Model/0923_skf_analysis/inference.py --config /data/ephemeral/home/Jungyeon/level1-imageclassification-cv-07/Model/0923_skf_analysis/config.json

# chmod +x /data/ephemeral/home/Jungyeon/level1-imageclassification-cv-07/Model/0923_skf_analysis/train.sh
# /data/ephemeral/home/Jungyeon/level1-imageclassification-cv-07/Model/0923_skf_analysis/train.sh