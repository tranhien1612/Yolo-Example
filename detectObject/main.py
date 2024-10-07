# from supervision.assets import download_assets, VideoAssets
# download_assets(VideoAssets.PEOPLE_WALKING) #download dataset

import cv2, math
import supervision as sv
from ultralytics import YOLO
from shapely.geometry import Polygon
from shapely.geometry.point import Point
import numpy as np

model = YOLO("yolov8n.pt")
image = cv2.imread("test1.jpg")

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

counting_regions = [
    {
        "name": "YOLOv8 Rectangle Region",
        "polygon": Polygon([(0, 0), (0, 300), (910, 300), (910, 0)]),  # Polygon points , anticlockwise
        "counts": 0,
        "region_color": (37, 255, 225),  # BGR Value
        "text_color": (255, 255, 255),  # Region Text Color
    },
]

def drawPolygon():
    for region in counting_regions:
        region_label = "Cnt: " + str(region["counts"])
        region_color = region["region_color"]
        region_text_color = region["text_color"]
        polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)

    print(region_label)
    cv2.putText(image, region_label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, 2)
    cv2.polylines(image, [polygon_coords], isClosed=True, color=region_color, thickness=2)

def box2object(results):
    for r in results:
        x1,y1,x2,y2=r.boxes.xyxy[0].round().int().numpy()
        conf = r.boxes.conf[0].float().numpy()
        cls = r.boxes.cls[0].int().numpy()
        cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2) 
        label=f'{classNames[cls]} {conf:.02f}'
        cv2.putText(image, label, (x1, y1),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)

        # Count object into polygon
        bbox_center = (x1 + x2) / 2, (y1 + y2) / 2 
        for region in counting_regions:
            if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                region["counts"] += 1
    drawPolygon()

box2object(results)

cv2.imshow("Image", image)
cv2.waitKey(0)
