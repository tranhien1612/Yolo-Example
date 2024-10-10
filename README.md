# OpenCV-Example

| Model       | Filenames                                                                                                      | Task                  |
| ----------- | -------------------------------------------------------------------------------------------------------------- | --------------------- |
| YOLOv8      | `yolov8n.pt` `yolov8s.pt` `yolov8m.pt` `yolov8l.pt` `yolov8x.pt`                                               | Detection             |
| YOLOv8-seg  | `yolov8n-seg.pt` `yolov8s-seg.pt` `yolov8m-seg.pt` `yolov8l-seg.pt` `yolov8x-seg.pt`                           | Instance Segmentation |
| YOLOv8-pose | `yolov8n-pose.pt` `yolov8s-pose.pt` `yolov8m-pose.pt` `yolov8l-pose.pt` `yolov8x-pose.pt` `yolov8x-pose-p6.pt` | Pose/Keypoints        |
| YOLOv8-obb  | `yolov8n-obb.pt` `yolov8s-obb.pt` `yolov8m-obb.pt` `yolov8l-obb.pt` `yolov8x-obb.pt`                           | Oriented Detection    |
| YOLOv8-cls  | `yolov8n-cls.pt` `yolov8s-cls.pt` `yolov8m-cls.pt` `yolov8l-cls.pt` `yolov8x-cls.pt`                           | Classification        |


[Train YOLOv8 on Custom Dataset ](https://learnopencv.com/train-yolov8-on-custom-dataset/)

The steps to train a YOLOv8 object detection model on custom data are:
1. Install YOLOv8 from pip
2. Create a custom dataset with labelled images
3. Use the yolo command line utility to run train a model
4. Run inference with the YOLO command line application

### How to Install YOLOv8

To install YOLOv8 from pip, use the following command:
```
pip install ultralytics
```

You can install the model from the source on GitHub using these commands:
```
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
pip install -e ultralytics
```

### Preparing a custom dataset for YOLOv8

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

### Train YOLOv8 on a custom dataset
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

### Validate with a new model
```
yolo task=detect mode=val model=runs/detect/yolov8n_v8_50e/weights/best.pt name=yolov8n_eval data=pothole_v8.yaml imgsz=1280
```

### Predict with a custom model
```
yolo task=detect \
mode=predict \
model=runs/detect/yolov8n_v8_50e/weights/best.pt \
conf=0.25 \
source=/test/images
```
