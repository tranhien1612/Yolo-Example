# OpenCV-Example

[Train YOLOv8 on Custom Dataset ](https://learnopencv.com/train-yolov8-on-custom-dataset/)

Download Dataset: 
```
wget https://www.dropbox.com/s/qvglw8pqo16769f/pothole_dataset_v8.zip?dl=1 -O pothole_dataset_v8.zip
unzip pothole_dataset_v8.zip
```
Create ```pothole_v8.yaml``` file:
```
path: pothole_dataset_v8/
train: 'train/images'
val: 'valid/images'
 
# class names
names: 
  0: 'pothole'
```

Install Libs:
```
pip install ultralytics
```

Train by CMD:
```
#train by yolo nano
yolo task=detect mode=train model=yolov8n.pt imgsz=1280 data=pothole_v8.yaml epochs=50 batch=8 name=yolov8n_v8_50e
#train by yolo small
yolo task=detect mode=train model=yolov8s.pt imgsz=1280 data=pothole_v8.yaml epochs=50 batch=8 name=yolov8s_v8_50e
#train by yolo medium
yolo task=detect mode=train model=yolov8m.pt imgsz=1280 data=pothole_v8.yaml epochs=50 batch=8 name=yolov8m_v8_50e
```
Train by code ```main.py```:
```
from ultralytics import YOLO
# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   data='pothole_v8.yaml',
   imgsz=640,
   epochs=10,
   batch=8,
   name='yolov8n_v8_50e'
```

Predict:
```
yolo task=detect mode=val model=runs/detect/yolov8n_v8_50e/weights/best.pt name=yolov8n_eval data=pothole_v8.yaml imgsz=1280
```
