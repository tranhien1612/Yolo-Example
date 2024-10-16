import cv2
import cvzone
import math
from ultralytics import YOLO

def main(model, cap):
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (980,740))

        results = model(frame)[0]
        for obj in results:     
            x1, y1, x2, y2 = obj.boxes.xyxy[0].round().int().numpy()
            conf = obj.boxes.conf[0].float().numpy()
            cls = obj.boxes.cls[0].int().numpy()
            # cv2.putText(frame, f'{cls}', (x1, y1 - 2), 0, 0.5, [0,255,0], thickness=1, lineType=cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = YOLO('../model/fall.pt')
    cap = cv2.VideoCapture('../data/fall/fall.mp4')
    main(model, cap)