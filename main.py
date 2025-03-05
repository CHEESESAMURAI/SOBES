from fastapi import FastAPI, File, UploadFile
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import shutil
import os

# Инициализация модели YOLOv8
model = YOLO("yolov8n.pt")  # Проверь, что этот файл скачан

# Список классов YOLO
COCO_CLASSES = model.model.names

app = FastAPI()

# Создаем папку для сохранения вырезанных объектов
os.makedirs("detections", exist_ok=True)


@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    """Обнаружение транспортных средств в изображении."""
    image_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Читаем изображение
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Ошибка чтения изображения"}

    # Запускаем YOLOv8
    results = model(image_path)

    detected_files = []
    for i, result in enumerate(results):
        for j, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            class_name = COCO_CLASSES[cls]

            if class_name in ['car', 'bus', 'truck', 'motorcycle', 'bicycle']:
                cropped_img = img[y1:y2, x1:x2]
                pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

                # Добавляем подпись
                draw = ImageDraw.Draw(pil_img)
                font = ImageFont.load_default()
                draw.text((10, 10), class_name, fill=(255, 0, 0), font=font)

                save_path = f"detections/{class_name}_{i}_{j}.png"
                pil_img.save(save_path)
                detected_files.append(save_path)

    return {"detected_images": detected_files}


# Инструкция по запуску в PyCharm
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# Dockerfile
DOCKERFILE_CONTENT = """
FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir torch torchvision torchaudio fastapi uvicorn ultralytics opencv-python-headless pillow
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# docker-compose.yml
DOCKER_COMPOSE_CONTENT = """
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
"""

# Команда для тестирования API
TEST_COMMAND = """
curl -X 'POST' \
  'http://localhost:8000/detect/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@TEST.JPG'
"""
