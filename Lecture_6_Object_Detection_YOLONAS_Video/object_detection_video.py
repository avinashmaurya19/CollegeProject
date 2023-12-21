import cv2
import torch
from super_gradients.training import models
import numpy as np
import math

cap = cv2.VideoCapture("..Video/traffic.mp4")

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = models.get('yolo_nas_s', pretrained_weights="coco").to(device)

count =0

while True:
    ret,frame = cap.read()
    count += 1
    if ret:
        result = list(model.predict(frame, conf=0.35))[0]
        bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
        confidences = result.prediction.confidence
        labels = result.prediction.label.tolist()
        for (bbox_xyxys, confidence, cls) in zip(bbox_xyxys, confidences, labels):
            bbox = np.array(bbox_xyxys)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            classname = int(cls)
            conf = math.ceil((confidence*100))/100
            label = f'{classname}{conf}'
            print("Frame N", count, "", x1, y1, x2, y2)
            t_size = cv2.getTextSize(label, 0, fontScale=1,thickness=2)
            c2=x1 + t_size[0], y1 - t_size[1] -3
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
        else:
            break

cv2.destroyAllWindows()
