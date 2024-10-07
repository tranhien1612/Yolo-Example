import cv2
import supervision as sv
from ultralytics import YOLO

image = cv2.imread("test.jpg")
model = YOLO("yolov8n.pt")
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



