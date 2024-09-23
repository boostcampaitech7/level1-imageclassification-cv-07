import requests
import json
import os
import sys
import json

class Slack():
    def __init__(self, token: str, user_id: str):
        self.token = token
        self.user_id = user_id


    def send_dm(self, message: str):
        url = 'https://slack.com/api/conversations.open'
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        data = {'users': [self.user_id]}
        
        # 채널 열기
        open_response = requests.post(url, headers=headers, data=json.dumps(data))
        open_data = open_response.json()
        if not open_data.get('ok'):
            raise Exception(f"Slack API Error: {open_data.get('error')}")
        
        channel_id = open_data['channel']['id']
        
        # 메시지 보내기
        message_url = 'https://slack.com/api/chat.postMessage'
        message_data = {
            'channel': channel_id,
            'text': message
        }
        message_response = requests.post(message_url, headers=headers, data=json.dumps(message_data))
        message_data = message_response.json()
        if not message_data.get('ok'):
            raise Exception(f"Slack API Error in chat.postMessage: {message_data.get('error')}")
        

def create_path(exp):
    py_dir_path = os.path.split(os.path.abspath(sys.argv[0]))[0]
    result_path = os.path.join(py_dir_path, exp['output_fold_path_format'].format(**exp))
    result_path = rename_if_exists(result_path)
    result_fold = os.path.split(result_path)[-1]
    exp_path = os.path.join(result_path, f"{result_fold}.json")

    exp['result_path'] = result_path
    exp['result_fold'] = result_fold
    exp['output_path'] = os.path.join(result_path, f"{result_fold}_predict.csv")
    exp['log_path'] = os.path.join(result_path, f"{result_fold}_log.csv")

    os.makedirs(result_path)
    with open(exp_path, 'w') as f:
        json.dump(exp, f, indent=4)

    return exp

def rename_if_exists(path):
    new_path = path

    if os.path.exists(new_path):
        i = 0
        while True:
            new_path = f'{path}_{i}'
            if not os.path.exists(new_path):
                break
            i += 1

            if 100000 < i:
                print("Fail to find a proper name")
                break
    
    return new_path
     
