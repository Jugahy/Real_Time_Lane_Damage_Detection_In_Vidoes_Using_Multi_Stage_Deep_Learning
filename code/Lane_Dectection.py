import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict
import time

import torch
# import logging

def image_to_tensor(image, size=(224, 224)):
    """이미지를 PyTorch 텐서로 변환하는 함수"""
    image = cv2.resize(image, size)  # 이미지를 지정된 크기로 리사이즈
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 이미지를 그레이스케일로 변환
    input_image = image / 255.0  # 픽셀 값을 0~1 사이로 정규화
    input_tensor = torch.reshape(torch.tensor(input_image, dtype=torch.float), (1, 1, size[0], size[1]))  # 텐서 생성
    return input_tensor

if __name__ == "__main__":
    # 설정 값 초기화
    view_img = True
    save_img = False

    setimg = './default_path/setimg2/'  # 기본 저장 경로
    setimg_m = './default_path/setimg2_mask/'  # 기본 마스크 저장 경로

    track_history = defaultdict(lambda: [])  # 궤적 기록

    # YOLO 모델 로드
    model = YOLO("./default_path/segment_best.pt")  # 기본 세그멘테이션 모델 경로
    names = model.names  # 클래스 이름

    # 분류 모델 로드
    model_classification = torch.load('./default_path/resnet18.pth', map_location=torch.device('cpu'))  # 기본 분류 모델 경로
    classes = ['A', 'B', 'C', 'F']  # 클래스 레이블
    classes_pen = {'A': (0, 255, 0), 'B': (255, 0, 0), 'C': (0, 255, 255), 'F': (0, 0, 255)}  # 색상
    model_classification.eval()  # 분류 모델 평가 모드 설정

    # 비디오 파일 경로 설정
    video_file = './default_path/video.mp4'  # 기본 비디오 파일 경로

    cap = cv2.VideoCapture(video_file)  # 비디오 캡처 객체 생성
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))  # 비디오 속성

    # ROI 설정
    minx_line = 0.25
    maxx_line = 0.8
    min_rows = int(h * minx_line)
    max_rows = int(h * maxx_line)

    # 객체 크기 기준 설정
    size_pixel = 12000
    size_seg_pixel = 1500

    # 텍스트 표시 설정
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2

    # FPS 계산 초기화
    fps_start_time = time.time()
    fps_counter = 0

    while True:
        ret, im0 = cap.read()  # 비디오 프레임 읽기
        if not ret:
            print("비디오 프레임이 비어 있거나 처리가 완료되었습니다.")
            break

        frame = im0.copy()

        annotator = Annotator(im0, line_width=1)

        # YOLO 모델을 사용하여 추적 수행
        results = model.track(im0, conf=0.1, persist=True, device='cpu')

        class_label_xy = []  # 클래스 좌표 저장
        mat = []  # 이미지 텐서 저장
        id_list = []  # 추적 ID 저장

        if results[0].boxes.id is not None and results[0].masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().tolist()
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, mask, track_id, cls in zip(boxes, masks, track_ids, clss):
                if len(mask) == 0:
                    print(box)
                    annotator.box_label(box, color=colors(track_id, True), label=names[int(cls)])
                    continue

                x, y, w1, h1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                size_p = (w1 - x) * (h1 - y)
                center_x = x + int((w1 - x) / 2)
                center_y = y + int((h1 - y) / 2)

                # ROI 필터링
                if center_y > max_rows or center_y < min_rows:
                    continue

                # 마스크 생성 및 크기 계산
                mask = mask.astype('int32')
                mask_image = np.zeros_like(frame[:, :, 0])
                try:
                    cv2.fillPoly(mask_image, [mask], (255, 255, 255))
                except Exception as e:
                    print("마스크 생성 중 오류:", e)

                size_seg = int(np.sum(mask_image) / 255)

                # 마스크 및 객체 크기 기준 필터링
                if size_p >= size_pixel and size_seg >= size_seg_pixel:
                    if save_img:
                        cv2.imwrite(setimg + names[int(cls)] + 'id' + str(track_id) + '_' + ".png", frame)
                        print(setimg + names[int(cls)] + 'id' + str(track_id) + ".png 저장 완료!")
                    if view_img:
                        cv2.rectangle(im0, (x, y), (w1, h1), color=colors(track_id, True), thickness=2)
                        annotator.seg_bbox(mask=mask,
                                           mask_color=colors(track_id, True),
                                           det_label=f"{names[int(cls)]} id:{track_id}")

        # FPS 계산 및 표시
        current_time = time.time()
        if (current_time - fps_start_time) >= 1:
            fps = fps_counter / (current_time - fps_start_time)
            fps_counter = 0
            fps_start_time = current_time
        cv2.putText(im0, f"FPS: {round(fps, 2)}", (20, 40), fontFace, fontScale, (255, 125, 0), thickness)
        fps_counter += 1

        # 화면 출력
        if view_img:
            cv2.imshow("Segmentation and Tracking", im0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()