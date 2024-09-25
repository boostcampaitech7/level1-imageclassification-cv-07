#!/bin/bash

# W&B 로그인을 먼저 수행해야 함 (첫 실행 시)
wandb login f959455a587ffb09e8677e2432dea61cd38cca5c
# config.json을 이용해 학습을 실행
python /data/ephemeral/home/Jihwan/level1-imageclassification-cv-07/Data/09_24_Resizing/train_again.py --config /data/ephemeral/home/Jihwan/level1-imageclassification-cv-07/Data/09_24_Resizing/config.json
python /data/ephemeral/home/Jihwan/level1-imageclassification-cv-07/Data/09_24_Resizing/inference.py --config /data/ephemeral/home/Jihwan/level1-imageclassification-cv-07/Data/09_24_Resizing/config.json

# chmod +x /data/ephemeral/home/Jihwan/level1-imageclassification-cv-07/Data/09_24_Resizing/train.sh
# /data/ephemeral/home/Jihwan/level1-imageclassification-cv-07/Data/09_24_Resizing/train.sh