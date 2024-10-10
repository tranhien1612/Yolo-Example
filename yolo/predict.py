from ultralytics import YOLO

model = YOLO("/home/maj/yolo/item/runs/my_run/weights/best.pt")

model.predict(source = "/home/maj/yolo/item/valid/images/bottles-of-global-soft-drink-brands-including-products-of-coca-cola-company-and-pepsico-WX0H4K_jpg.rf.4d0122f77473252d0b89cfa2611deca5.jpg", 
    project ="/home/maj/yolo/item/runs",
    name = "predict",
    show=True, save=True, 
    conf=0.6, 
    line_thickness = 2, 
    save_crop=True
    )# , hide_labels=True, hide_conf=True)
