import cv2, string
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
# License Plate 
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}
def license_complies_format(text):
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False
    
def format_license(text):
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_
   
def read_license_plate(reader, license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score
    return None, None
    
# License Plate 

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
        
            
