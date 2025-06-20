from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import shutil
import os
import cv2


app = FastAPI()
templates = Jinja2Templates(directory="")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    filename = os.path.join(UPLOAD_DIR, file.filename)
    with open(filename, "wb") as f:
        shutil.copyfileobj(file.file, f)
    money = []
    return templates.TemplateResponse("index.html", {"request": request, "money": money})
