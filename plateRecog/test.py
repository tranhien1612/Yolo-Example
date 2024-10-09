from ultralytics import YOLO
import numpy as np
import cv2
# from paddleocr import PaddleOCR
import easyocr

def box2object(detections, image):
    for r in detections:
        x1, y1, x2, y2, sorce = r
        if sorce > 0.5:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2) 
            label=f'{sorce:.02f}'
            cv2.putText(image, label, (int(x1), int(y1)),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
    
# Load Model
license_plate_detector = YOLO("license_plate_detector.pt") #("model.pt")

# Read frame
# frame = cv2.imread("test.jpg")

# license_plate_ = []
# license_plates = license_plate_detector(frame)[0]
# for license_plate in license_plates.boxes.data.tolist():
#     x1, y1, x2, y2, score, class_id = license_plate
#     license_plate_.append([x1, y1, x2, y2, score])

# box2object(license_plate_, frame)

# cv2.imshow("Image", frame)
# cv2.waitKey(0)

cnt = 0

reader = easyocr.Reader(['en'], gpu=False)
def readText(image):
    results = reader.readtext(image)
    ocr = ""
    conf = 0.2
    for result in results:
        if len(results) == 1:
            ocr = result[1]
        if len(results) >1 and len(results[1])>6 and results[2]> conf:
            ocr = result[1]
    
    return str(ocr)

# ocr = PaddleOCR(use_angle_cls=True, lang='en')
# def readText(img):
#     results = ocr.ocr(img, cls=True)
#     for line in results:
#         for word_info in line:
#             # Each word_info contains the bounding box, text, and confidence
#             box = word_info[0]  # Bounding box
#             text = word_info[1][0]  # Detected text
#             confidence = word_info[1][1]  # Confidence score
#             print(f'Text: {text}, Confidence: {confidence:.2f}')
    

while True:
    filename = f"test{cnt}.jpg"
    frame = cv2.imread(filename)
    if cnt > 1:
        frame = cv2.resize(frame, (1280, 720))
    license_plate_ = []
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        license_plate_.append([x1, y1, x2, y2, score])

        im = frame[int(y1):int(y2), int(x1):int(x2)]
        txt = readText(im)
        if score > 0.5:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2) 
            label=f'{txt}' #{score:.02f} 
            cv2.putText(frame, label, (int(x1), int(y1)),0, 1,[255,0,255], thickness=1,lineType=cv2.LINE_AA)

    # box2object(license_plate_, frame)

    cv2.imshow("Image", frame)
    
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cnt += 1

cv2.destroyAllWindows()
