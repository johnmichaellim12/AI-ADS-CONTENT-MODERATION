from ultralytics import YOLO
import os

# ✅ Load a pre-trained YOLOv8 model
model = YOLO("yolov8.pt")  # "n" = nano, fastest. Use "m" or "l" for more accuracy.

# ✅ Train the model
model.train(data="dataset/data.yaml", epochs=50, imgsz=640)



print(os.path.exists("C:/Users/NCTV_User_002/Desktop/pagsulay/dataset/images/train"))
print(os.path.exists("C:/Users/NCTV_User_002/Desktop/pagsulay/dataset/images/val"))

