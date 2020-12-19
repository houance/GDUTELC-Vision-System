import cv2
import numpy as np
import time


class YOLO:
    def __init__(self, yoloPath=1, gpu=0, confidence=0.5, threshold=0.3):
        self.path = self.chooseYoloVersion(yoloPath)
        self.colour, self.labels = self.colourGenerator()
        self.net = None
        self.ln = None
        self.yolov3Init(gpu)
        self.confidence = confidence
        self.threshold = threshold

    def chooseYoloVersion(self, yoloPath):
        if yoloPath == 1:
            cocoName = 'yolo/coco.names'
            yoloCfg = 'yolo/yolov3.cfg'
            yoloWeights = 'yolo/yolov3.weights'
            path = [cocoName, yoloCfg, yoloWeights]
        elif yoloPath == 2:
            cocoName = 'yolo/coco.names'
            yoloCfg = 'yolo/yolov3-tiny-prn.cfg'
            yoloWeights = 'yolo/yolov3-tiny-prn.weights'
            path = [cocoName, yoloCfg, yoloWeights]
        else:
            path = yoloPath
        return path

    def yolov3Init(self, gpu):
        if gpu == 0:
            self.net = cv2.dnn.readNetFromDarknet(self.path[1], self.path[2])
        else:
            self.net = cv2.dnn.readNetFromDarknet(self.path[1], self.path[2])
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def colourGenerator(self):
        labels = open(self.path[0]).read().strip().split('\n')
        np.random.seed(42)
        colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
        return colours, labels

    def predict(self, image, detectionFilter=-1, painted=1):
        imageNew = image.copy()
        imageNewHeight, imageNewWidth = imageNew.shape[:2]

        blob = cv2.dnn.blobFromImage(imageNew, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)

        boxes = []
        confidences = []
        classIDs = []
        detectionBoxes = []
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if isinstance(detectionFilter, list):
                    filterList = set(detectionFilter)
                    if classID + 1 not in filterList:
                        continue

                elif detectionFilter != -1:
                    if classID != detectionFilter - 1:
                        continue

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confidence:
                    # scale the bounding box coordinates back relative to
                    # the size of the imageNew, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([imageNewWidth, imageNewHeight, imageNewWidth, imageNewHeight])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in self.colour[classIDs[i]]]
                text = "{}: {:.4f}".format(self.labels[int(classIDs[i])], confidences[i])
                if painted:
                    cv2.rectangle(imageNew, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(imageNew, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                detectionBoxes.append([x, y, w, h, classIDs[i]])
        return imageNew, detectionBoxes
