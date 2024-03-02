import cv2
import torch
import numpy as np
import math
from sort import Sort

# Import YOLO model
from super_gradients.training import models

# Initialize YOLO model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.get('yolo_nas_s', pretrained_weights="coco").to(device)

# Initialize video capture
cap = cv2.VideoCapture("../Video/traffic_highway.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define lanes and corresponding limits
limit_up = [644, 413, 1271, 412]
limit_down = [3, 411, 596, 411]

# Initialize tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Initialize counts
total_count_up = []
total_count_down = []

# # Initialize traffic light durations
# green_light_duration = 10  # in seconds
# red_light_duration = 5  # in seconds

while True:
    ret, frame = cap.read()

    if ret:
        # Detect objects using YOLO
        detections = np.empty((0, 5))
        result = list(model.predict(frame, conf=0.35))[0]
        bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
        confidences = result.prediction.confidence
        labels = result.prediction.labels.tolist()

        for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
            bbox = np.array(bbox_xyxy)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            classname = int(cls)
            conf = math.ceil((confidence * 100)) / 100
            current_array = np.array([x1, y1, x2, y2, conf])
            detections = np.vstack((detections, current_array))

        # Update tracker
        results_tracker = tracker.update(detections)

        # Draw lane boundaries
        cv2.line(frame, (limit_up[0], limit_up[1]), (limit_up[2], limit_up[3]), (255, 0, 0), 5)
        cv2.line(frame, (limit_down[0], limit_down[1]), (limit_down[2], limit_down[3]), (255, 0, 0), 5)

        # Process each tracked object
        for result in results_tracker:
            if len(result) < 5:
                print("Unexpected number of values in result:", result)
                continue
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (85, 45, 255), 3)
            label = f'{int(id)}'
            # label = f'{int(result[-1])}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(frame, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, y1 - 2), 0, 1,
                        [255, 255, 255], thickness=1, lineType=cv2.INTER_AREA)

            # Count vehicles in respective lanes
            if limit_up[0] < cx < limit_up[2] and limit_up[1] - 15 < cy < limit_up[3] + 15:
                if result[-1] not in total_count_up:
                    total_count_up.append(result[-1])
            elif limit_down[0] < cx < limit_down[2] and limit_down[1] - 15 < cy < limit_down[3] + 15:
                if result[-1] not in total_count_down:
                    total_count_down.append(result[-1])

            # Draw object bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (85, 45, 255), 3)
            cv2.putText(frame, str(int(result[-1])), (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Control traffic lights based on vehicle count
        if len(total_count_up) > len(total_count_down):
            cv2.putText(frame, "Green Light - Up Lane", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Red Light - Down Lane", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Red Light - Up Lane", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Green Light - Down Lane", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display vehicle counts
        cv2.putText(frame, f"Up Lane: {len(total_count_up)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Down Lane: {len(total_count_down)}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
