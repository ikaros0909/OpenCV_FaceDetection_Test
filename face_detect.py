import sys
import numpy as np
import cv2
import math


model = 'opencv_face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'
config = 'opencv_face_detector/deploy.prototxt'
# model = 'opencv_face_detector/opencv_face_detector_uint8.pb'
# config = 'opencv_face_detector/opencv_face_detector.pbtxt'

# cap = cv2.VideoCapture(0) #카메라로 읽기
cap = cv2.VideoCapture('resource/jimi2525_49446_167751_inverse.mp4') #파일로 읽기
# cap = cv2.VideoCapture('resource/jimi2525_49446_167750.mp4') #파일로 읽기


if not cap.isOpened():
    print('Camera open failed!')
    sys.exit()

net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    sys.exit()

# cp = (src.shape[1] / 2, src.shape[0] / 2) # 영상의 가로 1/2, 세로 1/2
# cp = (150,150)
# rot = cv2.getRotationMatrix2D(cp, 20, 0.5) # 20도 회전, 스케일 0.5배

# net = cv2.warpAffine(src, rot, (0, 0))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
    net.setInput(blob)
    out = net.forward()

    detect = out[0, 0, :, :]
    (h, w) = frame.shape[:2]

    for i in range(detect.shape[0]):
        confidence = detect[i, 2]
        if confidence < 0.5:
            break

        x1 = int(detect[i, 3] * w)
        y1 = int(detect[i, 4] * h)
        x2 = int(detect[i, 5] * w)
        y2 = int(detect[i, 6] * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

        label = f'Face: {confidence:4.2f}'
        cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
