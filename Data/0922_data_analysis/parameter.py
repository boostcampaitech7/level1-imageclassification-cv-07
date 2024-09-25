import argparse
import json
from models.model_selector import ModelSelector

def main(config):
    # 모델 설정
    model_selector = ModelSelector(
        model_type="timm",
        num_classes=config.get('num_classes', 500),  # config에서 num_classes 가져오고, 없으면 기본값 500 사용
        model_name=config['model_name'],  
        pretrained=True
    )
    model = model_selector.get_model()

    # 총 파라미터 수와 학습 가능한 파라미터 수 계산
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 모델 정보와 파라미터 개수 출력
    print(f"Model: {config['model_name']}")
    print(model)  
    print(f"Total Parameters: {num_params:,}")  
    print(f"Trainable Parameters: {trainable_params:,}") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model parameter analysis using config file")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # config 파일을 읽어와 JSON 형식으로 변환
    with open(args.config, 'r') as f:
        config = json.load(f)

    # main 함수 실행
    main(config)

    
# python /data/ephemeral/home/Sojeong/level1-imageclassification-cv-07/sojeong_code/LV1/parameter.py --config /data/ephemeral/home/Sojeong/level1-imageclassification-cv-07/sojeong_code/LV1/config.json