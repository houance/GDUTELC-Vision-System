import cv2
import numpy as np
from priorbox import PriorBox
from utils import nms
import time


class faceDetectorModel:
    def __init__(self, method='haarCascades', gpu=0, confidence=0.7, threshold=0.3):
        self.gpu = gpu
        self.method = method
        self.init = 0
        self.detector = None
        self.pb = None
        self.detectorInit()
        self.confidence = confidence
        self.threshold = threshold

    def detectorInit(self):
        if self.method == 'haarCascades':
            if self.gpu == 0:
                self.detector = cv2.CascadeClassifier('faceDetect/haarcascade_frontalface_default.xml')
            elif self.gpu == 1:
                self.detector = cv2.cuda.CascadeClassifier_create('faceDetect'
                                                                  '/haarcascade_frontalface_default_cuda.xml')

        elif self.method == 'lbpCascades':
            if self.gpu == 0:
                self.detector = cv2.CascadeClassifier('faceDetect/lbpcascade_frontalface_improved.xml')
            elif self.gpu == 1:
                self.detector = cv2.cuda.CascadeClassifier_create('faceDetect/lbpcascade_frontalface_improved.xml')

        if self.method == 'yuNet':
            self.detector = cv2.dnn.readNet('faceDetect/YuFaceDetectNet_640.onnx')
            self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def CascadesDetector(self, frame):
        if self.gpu == 1:
            faces = []
            gpuFrame = cv2.cuda_GpuMat()
            gpuFrame.upload(frame)
            gpuMat = cv2.cuda.cvtColor(gpuFrame, cv2.COLOR_BGR2GRAY)
            objbuff = self.detector.detectMultiScale(gpuMat)
            facess = objbuff.download()
            if facess is None:
                facess = ()
            np.array(facess)
            for multipleFace in facess:
                for face in multipleFace:
                    faces.append(face)
            return faces
        elif self.gpu == 0:
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(grayFrame, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
            return faces

    def yuNetDetection(self, frame):
        if self.init == 0:
            frameWidth, frameHeight = frame.shape[:2]
            self.pb = PriorBox(input_shape=(640, 480), output_shape=(frameHeight, frameWidth))
            self.init = 1

        blob = cv2.dnn.blobFromImage(frame, size=(640, 480))
        outputNames = ['loc', 'conf', 'iou']
        self.detector.setInput(blob)
        loc, conf, iou = self.detector.forward(outputNames)
        dets = self.pb.decode(np.squeeze(loc, axis=0), np.squeeze(conf, axis=0), np.squeeze(iou, axis=0))
        idx = np.where(dets[:, -1] > self.confidence)[0]
        dets = dets[idx]

        if dets.shape[0]:
            facess = nms(dets, self.threshold)
        else:
            facess = ()
            return facess
        faces = np.array(facess[:, :4])
        faces = faces.astype(np.int)
        faceStartXY = faces[:, :2]
        faceEndXY = faces[:, 2:4]
        faceWH = faceEndXY - faceStartXY
        faces = np.hstack((faceStartXY, faceWH))
        # scores = facess[:, -1]
        return faces

    def predict(self, frame, painted=1):
        frameNew = frame.copy()
        faces = ()
        if self.method == 'haarCascades' or self.method == 'lbpCascades':
            faces = self.CascadesDetector(frameNew)
        elif self.method == 'yuNet':
            faces = self.yuNetDetection(frameNew)

        if painted:
            for (x, y, w, h) in faces:
                cv2.rectangle(frameNew, (x, y), (x + w, y + h), (0, 0, 255))

        return frameNew, faces
