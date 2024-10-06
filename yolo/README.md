

### Create env:
Open anaconda CMD
```
    conda -n <name> python=3.11
    conda activate <name>
```

### Install lib for env:
```
    pip install ultralytics
    pip install label-studio
```

### Using label-studio to create label for image
Start label-studio in cmd:
```
    lable-studio
```
Open website and login: http://localhost:8080/

Creat project, add image, create label and export

### Yolo tree folder
```
project
|----train
|    ----images
|    ----labels
|----val
|    ----images
|    ----labels
|----dataset.yaml
|----train.py
|----predict.py
```

### Create dataset.yaml:
```
    train: D:\test\train
    val: D:\test\val

    nc: 1
    names: ["cosy cake"]
```
nc: Number of class
names: array contain name of class

### Create train.py
```
from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(data = "dataset.yaml", 
            imgsz = 640, batch = 8, 
            epochs = 10, workers = 0, 
            device="cpu")
```
batch: Number of image in one train circle

epochs: Number of train circle

### Create predict.py
```
from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(source = "1.jpg", 
    show=True, save=True, 
    conf=0.6, 
    line_thickness = 1, 
    save_crop=True
    )
    # , hide_labels=True, hide_conf=True)
```