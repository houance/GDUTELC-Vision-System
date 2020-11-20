import cv2
import numpy as np
import time
from faceDetection import faceDetector


# cap = cv2.VideoCapture(0)
# start = time.time()
# detector = faceDetector(method='haarCascades', gpu=1)
# print(time.time() - start)
#
# while True:
#     ret, frame = cap.read()
#     frames = detector.predict(frame)
#     cv2.imshow('iamge', frames)
#     cv2.waitKey(1)

# detector = cv2.cuda.CascadeClassifier_create('faceDetect/lbpcascade_frontalface_improved.xml')
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     start = time.time()
#     gpuFrame = cv2.cuda_GpuMat()
#     gpuFrame.upload(frame)
#     gpuMat = cv2.cuda.cvtColor(gpuFrame, cv2.COLOR_BGR2GRAY)
#     objbuff = detector.detectMultiScale(gpuMat)
#     faces = objbuff.download()
#     print(time.time() - start)
#     for face in faces:
#         for (x, y, w, h) in face:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))
#     cv2.imshow('iamge', frame)
#     cv2.waitKey(1)


cap = cv2.VideoCapture(0)

detector = faceDetector(method='haarCascades', gpu=1)


while True:
    ret, frame = cap.read()
    frames, faces = detector.predict(frame)
    print(faces)
    cv2.imshow('image', frame)
    cv2.waitKey(1)

