from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data = "/home/maj/yolo/item/data.yaml", 
            project ="/home/maj/yolo/item/runs",
            name = "detect",
            imgsz = 640, batch = 9, 
            epochs = 100, workers = 0, 
            device="cpu")
