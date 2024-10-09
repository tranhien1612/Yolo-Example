# from supervision.assets import download_assets, VideoAssets
# download_assets(VideoAssets.PEOPLE_WALKING) #download dataset

import cv2
from ultralytics import YOLO
from shapely.geometry import Polygon

from func.drawArea import *

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

counting_regions = {
        "name": "YOLOv8 Rectangle Region",
        "polygon": Polygon([(0, 0), (0, imgSize[1] // 2), (imgSize[0], imgSize[1] // 2), (imgSize[0], 0)]),  # Polygon points , clockwise
        "counts": 0,
        "region_color": (37, 255, 225),  # BGR Value
        "text_color": (255, 255, 255),  # Region Text Color
        "thickness": 2,
    }


# counting_regions = {
#         "name": "YOLOv8 Line Region",
#         "polygon": Polygon([(0, 300), (imgSize[0], 300), (imgSize[0], 300), (0, 300)]), # Line (0, 200) - (1000, 200)
#         "counts": 0,
#         "region_color": (37, 255, 225),  # BGR Value
#         "text_color": (255, 255, 255),  # Region Text Color
#         "thickness": 2,
#     }

def main(results, image):
    draw = CustomDraw()

    for obj in results:
        draw.show_obj_inf(obj)
        draw.drawBox_each_object(image, obj, showLabel=True)
        draw.check_obj_in_area(obj, counting_regions, classId=2, minConf=0.80)

    draw.drawPolygon(image, counting_regions, showCnt=True)

    cv2.imshow("Image", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    image = cv2.imread("test1.jpg")
    image = cv2.resize(image, imgSize)

    model = YOLO("yolov8n.pt")
    results = model(image)[0]

    main(results, image)


