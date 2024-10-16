from ultralytics import YOLO
import cv2

classNames = ["red", "green", "", "yellow"]

def main(model, filepath):
    image = cv2.imread(filepath)
    results = model(image)[0]
    for result in results:
        x1, y1, x2, y2 = result.boxes.xyxy[0].round().int().numpy()
        conf = result.boxes.conf[0].float().numpy()
        cls = result.boxes.cls[0].int().numpy()
        p1 = (x1, y1)
        p2 = (x2, y2)
        if conf >= 0.0:
            cv2.rectangle(image, p1, p2, (255,0,0), 2)
            label = f"{classNames[cls]} {conf:.02f}"
            # label = f"{cls} {conf:.02f}"
            textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
            c2 = x1 + textSize[0], y1 - textSize[1] - 3
            cv2.rectangle(image, (x1, y1), c2, (255, 0, 0), -1)
            cv2.putText(image, label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)

    cv2.imshow('frame', image)
    cv2.waitKey(0)

def video_handle(model, filename):
    cap = cv2.VideoCapture(filename)
    while True:
        ret, image = cap.read()
        image = cv2.resize(image, (1280, 720))
        results = model(image)[0]
        for result in results:
            x1, y1, x2, y2 = result.boxes.xyxy[0].round().int().numpy()
            conf = result.boxes.conf[0].float().numpy()
            cls = result.boxes.cls[0].int().numpy()
            p1 = (x1, y1)
            p2 = (x2, y2)
            if conf >= 0.0:
                cv2.rectangle(image, p1, p2, (255,0,0), 2)
                label = f"{classNames[cls]} {conf:.02f}"
                textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + textSize[0], y1 - textSize[1] - 3
                cv2.rectangle(image, (x1, y1), c2, (255, 0, 0), -1)
                cv2.putText(image, label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)

        cv2.imshow("video", image)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = YOLO("../model/traffic_light.pt")
    main(model, "../data/traffic_light/test2.jpg")
    # video_handle(model, "../data/traffic_light/video1.mp4")