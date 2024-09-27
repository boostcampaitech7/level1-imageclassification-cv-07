import train
import inference
import json
import argparse
import os
from dotenv import load_dotenv
import utils.utils
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model using config file")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    exp_file_path = args.config

    with open(exp_file_path, 'r') as f:
        exp = json.load(f)

    load_dotenv() # load .env file
    SLACK_TOKEN = os.environ.get('SLACK_TOKEN')
    SLACK_USER_ID = os.environ.get('SLACK_USER_ID')
    WANDB_TOKEN = os.environ.get('WANDB_TOKEN')

    slack = utils.utils.Slack(SLACK_TOKEN, SLACK_USER_ID) # 슬랙 설정
    subprocess.call(f"wandb login {WANDB_TOKEN}", shell=True) # wandb 로그인

    config = {}
    config['train_data_dir'] = os.environ.get('train_data_dir')
    config['test_data_dir'] = os.environ.get('test_data_dir')
    config['data_info_file'] = os.environ.get('data_info_file')
    config['test_info_file'] = os.environ.get('test_info_file')
    config['wandb_project'] = os.environ.get('wandb_project')
    config['slack'] = slack
    exp = utils.utils.create_path(exp)
    
    # result_path를 포함하여 json파일 다시 기록하기
    with open(exp_file_path, 'w') as f:
        json.dump(exp, f, indent=4)

    train.main(exp, config)
    #inference.main(exp, config)