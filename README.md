# Real-Time Lane Damage Detection In Vidoes Using Multi-Stage Deep Learning
YOLOv8 모델을 활용해 영상 데이터에서 실시간으로 차선을 감지하고, 감지된 차선을 ResNet 모델로 분석하여 차선의 훼손도를 분류하는 시스템을 개발했습니다. 이를 통해 영상 데이터를 기반으로 차선의 훼손도를 실시간으로 평가할 수 있는 모델을 개발하였습니다.<br/>

Using the YOLOv8 model, we developed a system that detects lanes in real time from video data and analyzes the detected lanes with a ResNet model to classify the damage of lanes. Through this, we developed a model that can evaluate the damage of lanes in real time based on video data.

### Table of contents 

1. [Overview](#1️⃣-overview)
2. [Role](#2️⃣-role)
3. [YOLOv8](#3️⃣-yolov8)
4. [Process](#4️⃣-process)
5. [Structure](#5️⃣-structure)
6. [References](#6️⃣-references)
   

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


## 3️⃣ YOLOv8
### 3-1. What is the YOLOv8
➀ YOLOv8 Architecture
* YOLOv8은 YOLOv5와 유사한 Backbone을 사용하며, CSPLayer에 몇 가지 변경 사항이 적용되어 이제 C2f 모듈로 불립니다. C2f 모듈(두 개의 컨볼루션이 있는 크로스 스테이지 부분 병목)은 고수준 특징을 문맥 정보와 결합하여 탐지 정확도를 향상시킵니다.
* YOLOv8은 앵커 프리 모델을 사용하며, 객체성, 분류, 회귀 작업을 독립적으로 처리하기 위해 분리된 헤드를 갖추고 있습니다. 이 디자인은 각 브랜치(branch)가 자신의 작업에 집중할 수 있도록 하며 모델의 전반적인 정확도를 향상시킵니다. YOLOv8의 출력 층에서는 객체성 점수를 위한 활성화 함수로 시그모이드 함수를 사용하여 바운딩 박스가 객체를 포함할 확률을 나타냅니다. 클래스 확률을 위해 소프트맥스 함수를 사용하여 객체가 가능한 각 클래스에 속할 확률을 나타냅니다.
* YOLOv8은 CIoU [76]와 DFL [114] 손실 함수를 사용하여 바운딩 박스 손실과 분류 손실을 위한 이진 교차 엔트로피를 계산합니다. 이러한 손실 함수들은 특히 작은 객체를 처리할 때 객체 탐지 성능을 향상시켰습니다.
* YOLOv8은 또한 YOLOv8-Seg 모델이라고 불리는 의미 분할 모델을 제공합니다. 백본은 CSPDarknet53 특징 추출기이며, 전통적인 YOLO 넥 아키텍처 대신 C2f 모듈이 뒤따릅니다. C2f 모듈 뒤에는 두 개의 분할 헤드가 있으며, 이는 입력 이미지에 대한 의미 분할 마스크를 예측하는 방법을 학습합니다. 이 모델은 YOLOv8과 유사한 탐지 헤드를 가지고 있으며, 다섯 개의 탐지 모듈과 예측 층으로 구성됩니다. YOLOv8-Seg 모델은 다양한 객체 탐지 및 의미 분할 벤치마크에서 최첨단 결과를 달성하면서도 높은 속도와 효율성을 유지합니다.
  
![image](https://github.com/user-attachments/assets/15762f28-5c78-4dc3-8604-7090c47632a5)


### 3-2. How to Use YOLOv8 (Custom Data)
➀ Custom Data로 YOLOv88 모델을 학습하는 경우에는 Image / Annotation 으로 이루어진 Data를 준비해야 합니다.
  * Custom Data는 [Roboflow](https://public.roboflow.com/)에서 제공하는 Custom Data를 이용할 수 있고, 또는 직접 구축할 수 있습니다.
  * Custom Data 구축 시 이미지 데이터와 정답 데이터는 확장자를 제외한 파일 이름은 동일해야 합니다.
  * YOLOv8에서 Annotation 파일의 확장자는 반드시 .txt 여야 합니다.

<br/>

➁ YOLOv8으로 Custom Data를 학습하기 위해서는 YAML 파일이 필요합니다. (YAML은 아래의 두 가지를 반드시 포함해야 합니다.)
   * 이미지와 정답이 저장되어 있는 디렉토리 정보
   * Detection 하고 싶은 클래스 종류와 대응되는 각각의 이름
```YAML
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 7
names: ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
```

<br/>

➂ 위의 ➀, ➁를 통해 만든 데이터를 사전 학습된 yolov8n.pt에 Fine Tuning 하기 위한 코드 입니다.
```python
# pip install ultralytics

from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='mydata.yaml', epoch=10)

results = model.predict(source='/content/test/')
```

<br/>


## 4️⃣ Process
### 4-1. Data Introduction
Dareesoft로 부터 받은 고속도로, 일반도로에서 차량 주행 영상 729GB 데이터를 Nas에 받았습니다.<br/>
아래는 예시 데이터입니다.

![original](https://github.com/user-attachments/assets/8ba9cf4d-5fb6-44b7-94b5-9471c3c427c9)

<br/>

### 4-2. Obtain Data From Video
위의 코드에 YOLOv8 객체 탐지 및 분할(segmentation) 모델을 활용하여 Video에서 Lane 부분만 추출하였습니다.

![lane2](https://github.com/user-attachments/assets/0f595341-86a1-4159-8d74-b991773097f6)

➀ 원본 영상에서 차선 부분 Box

![image](https://github.com/user-attachments/assets/b7f820f2-693e-4b78-924e-ec3a245ac949)

➁ Segmentation Mask만 추출한 이미지

![image](https://github.com/user-attachments/assets/64247cc8-2a6c-4f6e-9fd8-1e56f7bf1a7f)

-> 이 데이터에는 차선 부분을 정확히 segmentation 하지 못하고 주변 도로도 포함하는 문제가 있습니다. <br/>
   (아래 절대감속을 예시로 설명드리겠습니다.)

차선 훼손도를 평가하는 모델에 차선 데이터를 학습할 때 차선이 아닌 다른 부분이 함께 있으면 학습에 안 좋은 영향을 줄 것이기 때문에 아래와 같은 전처리 방법을 참고하여 도로 부분은 제거하고 차선만 남을 수 있게 하였습니다.
* Otsu, Sobel, HSV, Canny, K-Means(각 픽셀의 0~255값의 분포를 통해 군집을 생성해 255에서 먼 군집을 0으로 변환)

<p>
  <img src="https://github.com/user-attachments/assets/517f1da7-8599-4b10-9a23-5d3a070c0ef3" width="640" height="320" alt="Image 1">
</p>

<p>
  <img src="https://github.com/user-attachments/assets/a6c78a79-7d5c-41c9-8a02-eca1708630b5" width="320" height="320" alt="Image 1">
  <img src="https://github.com/user-attachments/assets/2cd94c6b-508e-4202-bd9f-2fc5e562d6b9" width="320" height="320" alt="Image 2">
</p>

<br/>

### 4-3. Labeling For Lane Damage Evaluation
차선 훼손도에 따른 Labeling 된 데이터가 없기 때문에 아래와 같은 기준을 세워 A, B, C, F로 직접 Labeling 해주었습니다.

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/01ddd520-7d1b-499f-baf8-8538ab9a6d16" width="100" height="100" alt="A"><br>
      <span>A</span>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/fd3fd06a-ec3e-4e19-ac18-1711a3959e80" width="100" height="100" alt="B"><br>
      <span>B</span>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/695d3874-0141-4513-9c53-79f91c74afe5" width="100" height="100" alt="C"><br>
      <span>C</span>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/a1662d49-bad1-4517-9c39-612b63c3c19b" width="100" height="100" alt="F"><br>
      <span>F</span>
    </td>
  </tr>
</table>

* A : 균열 없이 완벽하게 횐색을 유지하고 있는가?
* B : 선의 형태로 균열이 존재하는가?
* C : 면의 형태로 균열이 존재하는가?
* F : 균열이 정상적인 부분보다 더 넓은가?

<br/>

### 4-4. Model For Lane Damage Evaluation
여러 Classification 모델을 학습하여 최적의 성능과 실시간 처리를 위한 빠른 분류가 가능한 모델을 찾았습니다.
아래의 모델을 Labeling한 데이터를 학습하여 성능을 평가해보았습니다.

* VGG-16
* Resnet-18
* Mobilenet V3 Large
* Mobilenet V3 Small

![image](https://github.com/user-attachments/assets/e21f0040-9030-4a63-bc91-ab7472e3b77a)

-> 다른 모델에 비해 Parameter 수가 적고 2번째로 높은 성능을 내는 **Mobilenet V3 Large Model**을 선택하였습니다.

<br/>

### 4-5. Result : Lane Segmentation & Classification in Real-Time Video

![lane 1](https://github.com/user-attachments/assets/b8252b5e-e8f9-4f0d-b8bc-30c2ddb1e0fe)



## 5️⃣ Structure
```
Real-Time Lane Damage Detection In Vidoes Using Multi-Stage Deep Learning

├── LICENSE
├── README.md
├── code
│   ├── Classification
│   │   ├── Nets_MobileNet_v3_small.ipynb
│   │   ├── mobilenet_v3_large.ipynb  
│   │   ├── resnet18.ipynb
│   │   ├── wide_resnet50_2.ipynb  
│   │   └── Nets_VGG16.ipynb
│   │   
│   ├── Lane_Classification.py
│   ├── Lane_Data_Preprocessing.ipynb
│   └── Lane_Detection.py

```



## 6️⃣ References
* https://docs.ultralytics.com/ko/models/yolov8/
* https://dongle94.github.io/paper/yolov5-8/
* https://www.youtube.com/watch?v=em_lOAp8DJE
