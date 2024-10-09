import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

'''
counting_regions = {
    "name": "Rectangle",
    "polygon": Polygon([(0, 0), (0, 0), (0, 0), (0, 0)]),  # Polygon points , anticlockwise
    "counts": 0,
    "region_color": (37, 255, 225),  # BGR Value
    "text_color": (255, 255, 255),  # Region Text Color
    "thickness": 2,
}

'''
class CustomDraw:

    def show_obj_inf(self, result):
        x1, y1, x2, y2 = result.boxes.xyxy[0].round().int().numpy()
        conf = result.boxes.conf[0].float().numpy()
        cls = result.boxes.cls[0].int().numpy()
        p1 = (x1, y1)
        p2 = (x2, y2)
        print(f"Box: {p1}, {p2} class_id: {cls} conf: {conf:.02f}")

    def check_obj_in_area(self, result, region, classId=0, minConf=0.0):
        x1,y1,x2,y2 = result.boxes.xyxy[0].round().int().numpy()
        conf = result.boxes.conf[0].float().numpy()
        cls = result.boxes.cls[0].int().numpy()
        bbox_center = (x1 + x2) / 2, (y1 + y2) / 2 

        if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))) and classId == cls and conf >= minConf:
            region["counts"] += 1
            return True

        return False

    def drawPolygon(self, img, region, showCnt=False):
        region_label = "Cnt: " + str(region["counts"])
        region_color = region["region_color"]
        region_text_color = region["text_color"]
        size = region["thickness"]
        polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)

        cv2.polylines(img, [polygon_coords], isClosed=True, color=region_color, thickness=size)
        if showCnt:
            cv2.putText(img, region_label, (polygon_coords[0][0] + 10, polygon_coords[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, thickness=size)
        
    def drawBox(self, img, p1, p2, label=""):
        cv2.rectangle(img, p1, p2, (255,0,0), 2)
        if label is not None and label != "":
            cv2.putText(img, label, p1 , cv2.FONT_HERSHEY_SIMPLEX, 0.7, [255,255,255], thickness=2)

    def drawBox_each_object(self, img, result, minConf = 0.5 ,showLabel=False):
        x1, y1, x2, y2 = result.boxes.xyxy[0].round().int().numpy()
        conf = result.boxes.conf[0].float().numpy()
        cls = result.boxes.cls[0].int().numpy()
        p1 = (x1, y1)
        p2 = (x2, y2)
        if conf >= minConf:
            cv2.rectangle(img, p1, p2, (255,0,0), 2)
            if showLabel == True:
                label = f"{cls}-{conf:.02f}"
                cv2.putText(img, label, p1 , cv2.FONT_HERSHEY_SIMPLEX, 0.7, [255,255,255], thickness=2)
        
            
