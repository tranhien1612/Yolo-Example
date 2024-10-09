from ultralytics import YOLO
from drawArea import *
import cv2
# from paddleocr import PaddleOCR
import easyocr

def readText(reader, image, minConf=0.2):
    results = reader.readtext(image)
    ocr = ""
    for result in results:
        if len(results) == 1:
            ocr = result[1]
        if len(results) > 1 and len(results[1])>6 and results[2] > minConf:
            ocr = result[1]
    return str(ocr)

# def readText(reader, img):
#     results = reader.ocr(img, cls=True)
#     for line in results:
#         for word_info in line:
#             # Each word_info contains the bounding box, text, and confidence
#             box = word_info[0]  # Bounding box
#             text = word_info[1][0]  # Detected text
#             confidence = word_info[1][1]  # Confidence score
#             print(f'Text: {text}, Confidence: {confidence:.2f}')
    
def image_handle(model, reader):
    filename = f"test0.jpg"
    image = cv2.imread(filename)

    draw = CustomDraw()
    results = model(image)[0]
    for obj in results:
        x1, y1, x2, y2, conf, cls = draw.drawBox_each_object(image, obj, showLabel=True)
        image_crop = image[y1: y2, x1: x2]
        txt = readText(reader, image_crop)
        print(txt)

    cv2.imshow("Image", image)
    cv2.waitKey(0)

def video_handle(filename, model, reader):
    cap = cv2.VideoCapture(filename)
    draw = CustomDraw()
    while True:
        ret, image = cap.read()
        image = cv2.resize(image, (1280, 720))
        results = model(image)[0]
        for obj in results:
            x1, y1, x2, y2, conf, cls = draw.drawBox_each_object(image, obj, minConf=0, showLabel=True)
            image_crop = image[y1: y2, x1: x2]
            txt = readText(reader, image_crop)
            print(txt)

        cv2.imshow("video", image)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

filename = "sample.mp4"
if __name__ == "__main__":
    model = YOLO("license_plate_detector.pt")
    reader = easyocr.Reader(['en'], gpu=False)
    # reader = PaddleOCR(use_angle_cls=True, lang='en')
    # image_handle(model, reader)
    video_handle(filename, model, reader)
