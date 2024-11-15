# Real-Time Lane Damage Detection In Vidoes Using Multi-Stage Deep Learning
YOLOv8 모델을 활용해 영상 데이터에서 실시간으로 차선을 감지하고, 감지된 차선을 ResNet 모델로 분석하여 차선의 훼손도를 분류하는 시스템을 개발했습니다. 이를 통해 영상 데이터를 기반으로 차선의 훼손도를 실시간으로 평가할 수 있는 모델을 개발하였습니다.<br/>

Using the YOLOv8 model, we developed a system that detects lanes in real time from video data and analyzes the detected lanes with a ResNet model to classify the damage of lanes. Through this, we developed a model that can evaluate the damage of lanes in real time based on video data.

### Table of contents 

1. [Overview](#1️⃣-overview)
2. [Role](#2️⃣-role)
3. [Process](#3️⃣-process)
4. [Structure](#4️⃣-structure)
5. [References](#5️⃣-references)
   

## 1️⃣ Overview
### 1-1. 개발 배경
WE-Meet을 통해 Dareesoft에서 연구 인턴으로 함께했습니다. Dareesoft는 도로에 포트홀(Pothole), 도로 균열, 낙하물, 로드킬 등과 같은 위험 요소 12가지를 실시간으로 감지하는 솔루션을 제공하는 기업입니다. 저희에게 주어진 문제는 새로운 도로 위 위험 요소인 차선이 손상되거나 긴 시간에 의해 지워져 운전자에게 혼란을 주고 차량 간 충돌을 야기할 수 있는 문제를 예방하는 것입니다. 이를위해 실시간으로 차선 훼손도를 평가하여 새로운 차선을 필요로 하는 위치를 제공하는 인공지능 모델을 개발하는 것을 목표로 하였습니다.

>### WE-Meet
>WE(Work Experience)-Meet 프로젝트는 산업계에서 문제해결 및 프로젝트 주제를 제시하고 대학생이 직접 프로젝트를 수행합니다. 기업과 대학은 학생에게 일경험 기회를 제공하여 우수인재를 발굴·검증하는 사회적 역할을 이행하고 대학생은 기업에 대한 이해와 적응력을 향상할 수 있습니다.

<br/>

### 1-2. 결과 미리보기
* Video 내에서 차선 부분 Detection
* 차선 훼손도 : A, B, C, F 중 각 차선에 해당하는 라벨과 그 라벨일 확률
* id : 각 차선의 고유한 Num
* S : Box Size
* SP : Number of split pixels
  
![lane 1](https://github.com/user-attachments/assets/b8252b5e-e8f9-4f0d-b8bc-30c2ddb1e0fe)


## 2️⃣ Role

|<img src="https://github.com/user-attachments/assets/bef1a11a-d69d-440a-9ed5-7c8f39548c5a" width="150" height="150"/>|<img src="https://github.com/user-attachments/assets/f9323ec6-0bfa-4dba-8589-4abb0948f2b7" width="150" height="150"/>|
|:-:|:-:|
|Jeong GangHyeon<br/>[@JUGAHY](https://github.com/JUGAHY)|Jang DaeHyeon<br/>[@JangDaeHyeon](https://github.com/JangDaeHyeon)|

### Jeong GangHyeon
* 
  
### Jang DaeHyeon
* 


## 3️⃣ Process
### 1-1. Data Introduction
Dareesoft로 부터 받은 고속도로, 일반도로에서 차량 주행 영상 729GB 데이터를 Nas에 받았습니다.<br/>
아래는 예시 데이터입니다.

![original](https://github.com/user-attachments/assets/8ba9cf4d-5fb6-44b7-94b5-9471c3c427c9)

  
### 1-2. How to Use YOLOv8 (Custom Data)
➀ Custom Data로 YOLOv88 모델을 학습하는 경우에는 Image / Annotation 으로 이루어진 Data를 준비해야 합니다.
  * Custom Data는 [Roboflow](https://public.roboflow.com/)에서 제공하는 Custom Data를 이용할 수 있고, 또는 직접 구축할 수 있습니다.
  * Custom Data 구축 시 이미지 데이터와 정답 데이터는 확장자를 제외한 파일 이름은 동일해야 합니다.
  * YOLOv8에서 Annotation 파일의 확장자는 반드시 .txt 여야 합니다.


```python
# pip install ultralytics

from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='mydata.yaml', epoch=10)

results = model.predict(source='/content/test/')
```
