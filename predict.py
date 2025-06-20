import cv2
from ultralytics import YOLO

tempImage = "test/images/3_20231106_172802-35-_jpg.rf.ab69f12ca0956df566ccf17c4ee29ae3.jpg"
model = YOLO("runs/detect/train/weights/last.pt")

im = cv2.imread(tempImage)
results = model.predict(source=im, save=True, save_txt=True)
print("="*30)
print(results)
print("="*30)