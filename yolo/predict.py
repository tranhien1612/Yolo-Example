from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(source = "1.jpg", 
    show=True, save=True, 
    conf=0.6, 
    line_thickness = 1, 
    save_crop=True
    )# , hide_labels=True, hide_conf=True)