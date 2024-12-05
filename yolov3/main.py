import cv2
import numpy as np
from PIL import Image

def run(frame, net): 
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), swapRB = True, crop = False)
    net.setInput(blob)
    
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    
    boxes = [] 
    confidences = []
    class_ids = []

    for output in layerOutputs: 
        for detection in output: 
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.2) 
    return indexes, boxes, class_ids, confidences

def boxing(frame, indexes, boxes, class_ids, confidences, classes, font):
    for i in indexes.flatten(): 
            x, y, w, h =  boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(frame, label + " " + confidence, (x, y+20), font, 2, (255, 255, 255), 4)
    return frame


def main(path):
    cap = cv2.VideoCapture(path)
    net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolo.cfg')

    classes = []
    with open('labels.txt', 'r') as f:
        classes = f.read().splitlines()

    while True:
        ret, frame = cap.read() 
        if ret:
            indexes = []
            boxes = []
            class_ids = []
            confidences = []
            indexes, boxes, class_ids, confidences = run(frame, net)

            if len(indexes) <= 0:
                continue
            elif len(indexes) > 0:
                frame = boxing(frame, indexes, boxes, class_ids, confidences, classes, cv2.FONT_HERSHEY_PLAIN)
            
            cv2.imshow('Output', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main("test.mp4")