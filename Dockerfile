FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копирование файлов проекта
COPY requirements.txt .
COPY main.py .

# Установка PyTorch отдельно для CPU версии
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Установка остальных зависимостей Python
RUN pip install --no-cache-dir -r requirements.txt

# Создание директории для моделей и установка прав
RUN mkdir -p /app/models && chmod 777 /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 