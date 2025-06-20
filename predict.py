import cv2
from ultralytics import YOLO

tempImage = ""
model = YOLO("<모델명>.pt")

im = cv2.imread(tempImage)
results = model.predict(source=im, save=True, save_txt=True)

print(results)