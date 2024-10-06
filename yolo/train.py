from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(data = "dataset.yaml", 
            imgsz = 640, batch = 8, 
            epochs = 1, workers = 0, 
            device="cpu")

