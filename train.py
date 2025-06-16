from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(data="data.yaml", epochs=100 , patience=10)

model.val()

model.export(format="onnx", dynamic=True)
