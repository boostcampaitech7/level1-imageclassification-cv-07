<div align='center'>
  <h1>🏆 Sketch Image Data Classification</h1>
  <img src='https://github.com/user-attachments/assets/bcb2cbdc-acb6-4ee4-815f-92089903456e'/>
</div><br>

**Goal :** 주어진 스케치 데이터를 활용하여 모델을 제작하고 제공된 테스트 세트의 각 이미지에 대해 올바른 레이블 예측 <br>
**Data :** 원본 ImageNet Sketch 데이터셋 중 상위 500개 객체(class)의 25,035개의 이미지 데이터

## [1] Project Overview
### ⏲️ Timeline (09/03 - 09/26)
<img src='https://github.com/user-attachments/assets/5bf0cf9c-1210-4234-a26c-8e29abe83441' width="70%"/>

1. EDA 및 baseline code 분석
2. Baseline model 선정
3. 모듈화 및 협업 툴 추가
4. Baseline model 일반화 성능 개선
5. 결과 분석

### 🥇 최종 결과
- 최종 리더보드 순위 캡쳐
- 랩업 리포트 링크

## [2] Environment
- OS : Linux-5.4.0
- GPU : Tesla V100 (32GB)
- Python Version: 3.10.13
- IDE: Visual Studio Code
- Tool : Github, Slack, Notion, Zoom
- Experiment Tracking: Weights and Biases (WandB)

## [3] File Tree
```
  ├─Report : report file
  ├─Model : final model code
    ├─Model.txt
    ├─datasets
      ├─custom_dataset.py
      ├─transform.py
    ├─models
      ├─model_selector.py
    ├─utils
      ├─train_utils.py
    ├─config.json
    ├─inference.py
    ├─inference.sh
    ├─train.py
    ├─train.sh
  ├─README.md
```
## [4] Project Workflow
1. EDA 및 baseline code 분석
   - 데이터는 Traindata 15,021개와 Private&Public Testdata 10,014개로 구성
   - Traindata : 15021개의 항목과 3개의 컬럼(class_name, image_path, target)으로 구성
   - Testdata : 10014개의 항목과 1개의 컬럼(image_path)으로 구성
   - 500개의 클래스가 있고 각 클래스마다 29~31개의 데이터로 구성
   - 분석한 데이터 특징
     -  흑백 이미지가 많지만 컬러 이미지도 존재
     -  스케치 선의 두께, 이미지 크기, 이미지의 해상도가 다양함
     -  한 이미지 안에 여러 개체가 들어있는 이미지도 존재
     -  정면, 측면 혹은 뒤집어진 이미지도 존재
   -  기본 baseline_code 모델의 정확도는 약 68.4%로 확인
2. Baseline model 선정
   - 다양한 backbone 모델 실험 후 최종적으로 가장 높은 88.3%의 public score를 달성한 **Coatnet_3_rw_224**를 baseline model로 선정
3. 모듈화 및 협업 툴 추가
   - Weights and Biases (WandB) 사용
   - github repository와 로컬 작업 공간 연결
   - tmux 사용
   - slack api 활용 모델 학습 완료 알림 자동화
4. Baseline model 일반화 성능 개선
   1) Data Augmentation
   2) K-fold
   3) Ensemble
   4) Other Experiments
5. 최종 결과 분석



## [5] Final Model Architecture

<div align='center'>
  <h1>Team Memebers</h1>
  <h3>럭키비키🍀</h3>
  <table>
    <tr>
      <td align="center" valign="top" width="150px"><a href="https://github.com/jinlee24"><img src="https://avatars.githubusercontent.com/u/137850412?v=4" ></a>이동진</td>
      <td align="center" valign="top" width="150px"><a href="https://github.com/stop0729"><img src="https://avatars.githubusercontent.com/u/78136790?v=4" ></a>정지환</td>
      <td align="center" valign="top" width="150px"><a href="https://github.com/yjs616"><img src="https://avatars.githubusercontent.com/u/107312651?v=4" ></a>유정선</td>
      <td align="center" valign="top" width="150px"><a href="https://github.com/sng-tory"><img src="https://avatars.githubusercontent.com/u/176906855?v=4" ></a>신승철</td>
      <td align="center" valign="top" width="150px"><a href="https://github.com/Soojeoong"><img src="https://avatars.githubusercontent.com/u/100748928?v=4" ></a>김소정</td>
      <td align="center" valign="top" width="150px"><a href="https://github.com/cyndii20"><img src="https://avatars.githubusercontent.com/u/90389093?v=4"></a>서정연</td>
    </tr>
    <tr>
      <td valign="top"> <!-- 동진 -->
          <ul>
              <li>각자 역할</li>
          </ul>
      </td>
      <td valign="top"> <!-- 지환 -->
          <ul>
              <li>각자 역할</li>
          </ul>
      </td>
       <td valign="top"> <!-- 정선 -->
          <ul>
              <li>각자 역할</li>
          </ul>
      </td>
       <td valign="top"> <!-- 승철 -->
          <ul>
              <li>각자 역할</li>
          </ul>
      </td>
       <td valign="top"> <!-- 소정 -->
          <ul>
              <li>각자 역할</li>
          </ul>
      </td>
       <td valign="top"> <!-- 정연 -->
          <ul>
              <li>각자 역할</li>
          </ul>
      </td>
    </tr>
  </table>
</div>
