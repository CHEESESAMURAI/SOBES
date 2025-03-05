# 1. Используем официальный образ Python 3.9
FROM python:3.9

# 2. Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# 3. Устанавливаем системные зависимости (ВАЖНО)
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# 4. Копируем файл с зависимостями
COPY requirements.txt /app/requirements.txt

# 5. Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# 6. Копируем все файлы проекта
COPY . /app

# 7. Запускаем FastAPI сервер
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
