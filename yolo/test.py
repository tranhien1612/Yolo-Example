# from supervision.assets import download_assets, VideoAssets
# download_assets(VideoAssets.PEOPLE_WALKING) #download dataset

import cv2, math
import supervision as sv
from ultralytics import YOLO
import numpy as np

model = YOLO("/home/maj/yolo/item/runs/detect/weights/best.pt")
image = cv2.imread("/home/maj/yolo/item/valid/images/test.jpg")

results = model(image)[0]

detections = sv.Detections.from_ultralytics(results)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence in zip(detections['class_name'], detections.confidence)
]

annotated_image = box_annotator.annotate(scene=image, detections=detections) #Create box cover object
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels) #Create label in object

cv2.imshow("yolo", annotated_image)
cv2.waitKey(0)
