# from supervision.assets import download_assets, VideoAssets
# download_assets(VideoAssets.PEOPLE_WALKING) #download dataset

import cv2, math
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
image = cv2.imread("test.jpg")

results = model(image)[0]
'''---------------------Using Lib---------------------'''
# detections = sv.Detections.from_ultralytics(results)

# box_annotator = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator()
# labels = [
#     f"{class_name} {confidence:.2f}"
#     for class_name, confidence in zip(detections['class_name'], detections.confidence)
# ]

# annotated_image = box_annotator.annotate(scene=image, detections=detections) #Create box cover object
# annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels) #Create label in object

# cv2.imshow("yolo", annotated_image)
# cv2.waitKey(0)

'''---------------------Not Use Lib---------------------'''
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# for r in results:
#     print(f"{r.boxes.xyxy[0]} {r.boxes.conf[0]} {r.boxes.cls[0]}")
#     # x1,y1,x2,y2=r.boxes.xyxy[0]
#     # conf=math.ceil((r.boxes.conf[0]*100))/100
#     # cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,255),3)

for r in results:
    boxes = r.boxes
    for box in boxes:
        x1,y1,x2,y2=box.xyxy[0]
        x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
        conf=math.ceil((box.conf[0]*100))/100
        cls=int(box.cls[0])
        print(x1, y1, x2, y2,conf, cls)

        cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,255),3) #Draw Box

        class_name=classNames[cls]
        label=f'{class_name}{conf}'
        t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]

        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(image, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # Rectangle contain text, filled 
        cv2.putText(image, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
cv2.imshow("Image", image)
cv2.waitKey(0)
