import cv2
from ultralytics import YOLO

tempImage = "test/images/test01.jpg"
model = YOLO("runs/detect/train/weights/last.pt")

im = cv2.imread(tempImage)
results = model.predict(source=im, save=True, save_txt=True)
print("="*30)
print(results)
print("="*30)