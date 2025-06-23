from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import os
import cv2
from ultralytics import YOLO


app = FastAPI()
app.mount("/runs", StaticFiles(directory="runs"), name="runs")

templates = Jinja2Templates(directory="templates")
UPLOAD_DIR = "runs/detect/latest"

mConvert = [10,50,100,500,1000,5000,10000,50000]
threshold = 0.8

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "flag":0})

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    filename = os.path.join(UPLOAD_DIR, file.filename)
    with open(filename, "wb") as f:
        shutil.copyfileobj(file.file, f)
    im = cv2.imread(UPLOAD_DIR + "/" + file.filename)

    model = YOLO("usingModel/last.pt")

    results = model.predict(source=im, save=True, project='runs/detect', name="latest", exist_ok=True)

    money = [0, 0, 0, 0, 0, 0, 0, 0]
    print("=" * 30)
    for result in results:
        boxes = result.boxes
        class_ids = boxes.cls  # tensor([4., 5., 6., ...])
        confidences = boxes.conf  # tensor([0.92, 0.87, ...])

        for cls_id, conf in zip(class_ids, confidences):
            print(f"인덱스 : {int(cls_id.item())}, 신뢰도 : {float(conf.item()):.2f}")
            if conf.item() > threshold:
                money[int(cls_id.item())] = money[int(cls_id.item())] + 1
    print("=" * 30)
    total = 0
    print(money)
    for idx, value in enumerate(money):
        total += mConvert[idx] * value
    print(total)
    money_data = list(zip(mConvert, money))
    return templates.TemplateResponse("index.html", {"request": request, "total":total, "money_data": money_data, "flag":1})
