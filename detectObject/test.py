import cv2, math
import supervision as sv
from ultralytics import YOLO
from shapely.geometry import Polygon
from shapely.geometry.point import Point
import numpy as np

imgSize = (1280, 720)
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
        "polygon": Polygon([(0, 0), (0, imgSize[1] // 2), (imgSize[0], imgSize[1] // 2), (imgSize[0], 0)]),  # Polygon points , anticlockwise
        "counts": 0,
        "region_color": (37, 255, 225),  # BGR Value
        "text_color": (255, 255, 255),  # Region Text Color
    },
]

def drawPolygon(image):
    for region in counting_regions:
        region_label = "Cnt: " + str(region["counts"])
        region_color = region["region_color"]
        region_text_color = region["text_color"]
        polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)

    cv2.putText(image, region_label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, 2)
    cv2.polylines(image, [polygon_coords], isClosed=True, color=region_color, thickness=2)

def box2object(results, image):
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

    drawPolygon(image)


def main():
    cap = cv2.VideoCapture(0)
    model = YOLO("yolov8n.pt")
    while True:
        ret, image = cap.read()
        results = model(image)[0]
        box2object(results, image)

        cv2.imshow("yolo", image)
        if cv2.waitKey(30) == 27:
            break

if __name__ == "__main__":
    main()
