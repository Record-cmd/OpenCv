# main.py

from picamera2 import Picamera2
import cv2
from person_counter import YOLODetector, CentroidTracker, LineCounter

def main():
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (1280, 720) #카메라 해상도설정
    picam2.preview_configuration.main.format = "RGB888" #포맷을 RGB로 설정
    picam2.configure("preview") #미리보기 설정 적용
    picam2.start()

    #객체 생성
    detector = YOLODetector()
    tracker = CentroidTracker()
    counter = LineCounter(line_x=640) #화면 중간에 기준선 좌표 설정

    print("System initialized. Press 'q' to quit.")

    while True:
        frame = picam2.capture_array() #한 프레임을 배열로 저장

        if frame.shape[2] == 4: #이미지의 채널수를 나타냄 만약 채널수가 4개라면
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)#RGBA를 RGB로 변환함

        detections = detector.detect(frame) #이미지채널을 변환한 프레임을 detect함수로 매개변수로 전달함
        objects = tracker.update(detections) #중심좌표리스트를 매개변수로긴후 움직임을 추적함
        counter.update(objects) #움직임을 기반으로 In out 판단

        for object_id, (cx, cy) in objects.items(): #모든 객체에 대해
            cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1) #객체의 중심에 파란점 표시
            cv2.putText(frame, str(object_id), (cx + 5, cy - 5), #객체옆 고유 id 표시
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)#putText(이미지대상, 텍스트, 위치, 글꼴, 글자크기, 색상, 두께)

        counter.draw(frame) #선, in, out 출력

        cv2.imshow("People Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Program ended.")

if __name__ == "__main__":
    main()

#==================================================================

# person_counter.py
import cv2
import numpy as np
from ultralytics import YOLO
from collections import OrderedDict
import math

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect(self, frame):
        input_frame = cv2.resize(frame, (320, 240)) #frame의 사이즈를 320x240으로 리사이즈함
        results = self.model(input_frame, verbose=False)[0] #사전에 학습된 self.model에 프레임을 넣어서 객체탐지를 수행함
        detections = []
        h_ratio = frame.shape[0] / 320 # 원본 이미지 크기에 맞게 되돌리는 게 목적입니다.
        w_ratio = frame.shape[1] / 240
        
        for result in results.boxes.data: #탐지 결과를 반복하여  사람만 필터링함
            x1, y1, x2, y2, score, cls = result
            if int(cls) == 0:  # person
                cx = int(((x1 + x2) / 2) * w_ratio) #중심좌표 계산
                cy = int((y1 + 0.6 * (y2 - y1)) * h_ratio)#중심좌표 계산
                detections.append((cx, cy)) #중심좌표 저장 여기에서는 보통 사람의 상체쯤
        print(f"Detected persons: {len(detections)}") #감지된 사람수 출력
        return detections #중심좌표리스트 반환


class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.max_distance = max_distance

    def update(self, detections):#중심좌표리스트를 매개변수로 전달받음
        updated_objects = OrderedDict()
        for cx, cy in detections:#각 탐지된 객체의 중심좌표를 각각 x,y로 나누어 반복함
            matched = False
            for obj_id, (x, y) in self.objects.items():
                distance = math.hypot(cx - x, cy - y) #이전 프레임의 객체와 유클리드 거리를 계산함
                if distance < self.max_distance: #거리조건을 만족하면 같은객체로 간주한다.
                    updated_objects[obj_id] = (cx, cy) #같은 id로 좌표를 저장함
                    matched = True
                    break
            if not matched:
                updated_objects[self.next_object_id] = (cx, cy)#가까운 객체가없으면 새로운 다음 id부여
                self.next_object_id += 1
        self.objects = updated_objects #오브젝트업데이트
        return self.objects

class LineCounter:
    def __init__(self, line_x=320):
        self.line_x = line_x #세로선 위치
        self.count_in = 0 #들어온사람수
        self.count_out = 0 #나간 사람수
        self.tracked = {}

    def update(self, objects):
        for object_id, (cx, cy) in objects.items():
            prev_cx = self.tracked.get(object_id, None)
            if prev_cx is not None:
                if prev_cx < self.line_x <= cx:
                    self.count_in += 1 #왼쪽에서 오른쪽으로 넘어감(들어옴)
                    print(f"ID {object_id} crossed line In")
                elif prev_cx > self.line_x >= cx:
                    self.count_out += 1#오른쪽 -> 왼쪽(나감)
                    print(f"ID {object_id} crossed line Out")
            self.tracked[object_id] = cx #현재 좌표를 저장 다음프레임과 비교

    def draw(self, frame): #선, 입장한사람 퇴장한사람 표시
        h, w = frame.shape[:2]
        cv2.line(frame, (self.line_x, 0), (self.line_x, h), (0, 255, 255), 2)
        cv2.putText(frame, f"In: {self.count_in}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Out: {self.count_out}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

