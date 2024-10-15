from ultralytics import YOLO
from drawArea import *
import cv2, torch

#https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2023-09-15--yolo8-tracking-and-ocr/2023-09-15/
#https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/model/4

classNames = ["Hardhat", "Mask", "", "No-Mask", "No-SafetyVest", "Person", "Safety-Cone", "Safety-Vest", "Machinery", "Vehicle"]

def image_handle(filename, model):
    image = cv2.imread(filename)
    # image = cv2.resize(image, (640, 512))
    draw = CustomDraw()
    results = model(image)[0]
    for obj in results: #5: person, 0: hat, 1: mask, 7: coat
        x1, y1, x2, y2, conf, cls = draw.drawBox_each_object(image, obj, classNames, showLabel=True)

    cv2.imshow("Image", image)
    cv2.waitKey(0)

def video_handle(filename, model):
    cap = cv2.VideoCapture(filename)
    draw = CustomDraw()
    while True:
        ret, image = cap.read()
        image = cv2.resize(image, (1280, 720))
        results = model(image)[0]

        for obj in results:
            x1, y1, x2, y2, conf, cls = draw.drawBox_each_object(image, obj, classNames, showLabel=True)

        cv2.imshow("video", image)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

videofilename = "../data/construction/video2.mp4"
imgfilename = "../data/construction/test.jpg"

if __name__ == "__main__":

    print("CUDA Available: ", torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    if device == 'cuda':
        torch.cuda.set_device(0) #set the GPU device before initializing the YOLOv8 model.
        # model = YOLO("../model/ppe_construction.pt", device='gpu')

    model = YOLO("../model/ppe.pt").to(device)

    image_handle(imgfilename, model)
    # video_handle(videofilename, model)

