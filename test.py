from multiprocessing import Process
from YOLOv3 import YOLO
import cv2

from faceDetection import faceDetectorModel


objectDetector = YOLO(gpu=1)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame, boxes = objectDetector.predict(frame)
cap.release()
