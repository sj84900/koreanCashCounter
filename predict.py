import cv2
from ultralytics import YOLO

imageLoot = "test/images/"
imageName = "test01.jpg"
model = YOLO("runs/detect/train/weights/last.pt")

im = cv2.imread(imageLoot + imageName)
results = model.predict(source=im, save=True, save_txt=True)
print("="*30)
for result in results:
    boxes = result.boxes
    class_ids = boxes.cls  # tensor([4., 5., 6., ...])
    confidences = boxes.conf  # tensor([0.92, 0.87, ...])

    for cls_id, conf in zip(class_ids, confidences):
        print(f"인덱스 : {int(cls_id.item())}, 신뢰도 : {float(conf.item()):.2f}")
print("="*30)