import cv2, re
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
def paddle_ocr(reader, frame, x1, y1, x2, y2):
    frame = frame[y1:y2, x1: x2]
    result = reader.ocr(frame, det=False, rec = True, cls = False)
    text = ""
    for r in result:
        #print("OCR", r)
        scores = r[0][1]
        if np.isnan(scores):
            scores = 0
        else:
            scores = int(scores * 100)
        if scores > 60:
            text = r[0][0]
    pattern = re.compile('[\W]')
    text = pattern.sub('', text)
    text = text.replace("???", "")
    text = text.replace("O", "0")
    text = text.replace("粤", "")
    return str(text)

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

    def drawBox_each_object(self, img, result, classNames=None, minConf = 0.5, showLabel=False):
        x1, y1, x2, y2 = result.boxes.xyxy[0].round().int().numpy()
        conf = result.boxes.conf[0].float().numpy()
        cls = result.boxes.cls[0].int().numpy()
        p1 = (x1, y1)
        p2 = (x2, y2)
        if conf >= minConf:
            cv2.rectangle(img, p1, p2, (255,0,0), 2)
            if showLabel == True:
                if classNames is not None:
                    label = f"{classNames[cls]} {conf:.02f}"
                else:
                    label = f"{cls} {conf:.02f}"
                # cv2.putText(img, label, p1 , cv2.FONT_HERSHEY_SIMPLEX, 0.7, [255,255,255], thickness=2)

                textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + textSize[0], y1 - textSize[1] - 3
                cv2.rectangle(img, (x1, y1), c2, (255, 0, 0), -1)
                cv2.putText(img, label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)

        return x1, y1, x2, y2, conf, cls
        
            
