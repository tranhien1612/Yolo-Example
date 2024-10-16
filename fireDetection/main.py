from ultralytics import YOLO
import cvzone
import cv2
import math


# Reading the classes
def main(model, cap):
    classnames = ['fire', 'smoke', 'smoke']
    while True:
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (640, 480))
        result = model(frame, stream=True)

        # Getting bbox,confidence and class names information to work with
        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                # print(f"conf: {confidence} class_id: {Class}")
                if confidence > 20:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    label = f'{classnames[Class]} {confidence}%'
                    textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                    c2 = x1 + textSize[0], y1 - textSize[1] - 3
                    cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)

                    # cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                    #                 scale=1.5, thickness=2)

        cv2.imshow('frame', frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = YOLO('../model/fire_smoke_yolov8.pt')
    cap = cv2.VideoCapture('../data/fire/fire.mp4')
    main(model, cap)


