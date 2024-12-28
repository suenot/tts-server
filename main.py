from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import io
import numpy as np
from fastapi.responses import StreamingResponse
import requests
import os
from pydantic import BaseModel
import logging
import re

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSRequest(BaseModel):
    text: str
    voice: str = "female"  # Добавляем выбор голоса: "female" или "male"

def split_by_language(text: str):
    # Разбиваем текст на предложения
    sentences = re.split('([.!?]+)', text)
    result = []
    current = ''
    
    # Объединяем предложения с их знаками препинания
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            current = sentences[i] + sentences[i+1]
        else:
            current = sentences[i]
            
        if current.strip():
            # Определяем язык по наличию кириллицы
            is_russian = bool(re.search('[а-яА-ЯёЁ]', current))
            result.append((current.strip(), 'ru' if is_russian else 'en'))
    
    # Добавляем последнее предложение, если оно есть
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        is_russian = bool(re.search('[а-яА-ЯёЁ]', sentences[-1]))
        result.append((sentences[-1].strip(), 'ru' if is_russian else 'en'))
    
    return result

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загружаем модели
device = torch.device('cpu')

# Проверяем наличие русской модели или скачиваем её
ru_model_path = "model_ru.pt"
if not os.path.isfile(ru_model_path):
    logger.info("Downloading Russian model...")
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
                                 ru_model_path)
    logger.info("Russian model downloaded successfully")

# Проверяем наличие английской модели или скачиваем её
en_model_path = "model_en.pt"
if not os.path.isfile(en_model_path):
    logger.info("Downloading English model...")
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/en/v3_en.pt',
                                 en_model_path)
    logger.info("English model downloaded successfully")

logger.info("Loading models...")
ru_model = torch.package.PackageImporter(ru_model_path).load_pickle("tts_models", "model")
en_model = torch.package.PackageImporter(en_model_path).load_pickle("tts_models", "model")
ru_model.to(device)
en_model.to(device)
logger.info("Models loaded successfully")

# Определяем похожие голоса для каждого языка
VOICE_MAPPING = {
    "female": {
        "ru": "kseniya",     # Женский голос для русского
        "en": "en_1"      # Наиболее похожий женский голос для английского
    },
    "male": {
        "ru": "aidar",     # Мужской голос для русского
        "en": "en_0"       # Наиболее похожий мужской голос для английского
    }
}

sample_rate = 48000

def generate_audio_for_text(text: str, lang: str, voice_type: str = "female"):
    model = ru_model if lang == 'ru' else en_model
    speaker = VOICE_MAPPING[voice_type][lang]
    return model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate)

def concatenate_audio(audio_list):
    if not audio_list:
        return None
    return torch.cat(audio_list, dim=0)

@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    try:
        logger.info(f"Received text: {request.text}")
        
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        # Разбиваем текст на части по языкам
        text_parts = split_by_language(request.text)
        logger.info(f"Split into parts: {text_parts}")

        # Генерируем аудио для каждой части
        audio_parts = []
        for text, lang in text_parts:
            logger.info(f"Generating audio for '{text}' ({lang}) with voice type {request.voice}")
            audio = generate_audio_for_text(text, lang, request.voice)
            audio_parts.append(audio)

        # Объединяем все части
        logger.info("Concatenating audio parts...")
        audio = concatenate_audio(audio_parts)
        
        if audio is None:
            raise HTTPException(status_code=400, detail="No valid text to synthesize")

        logger.info("Converting to WAV format...")
        audio_np = audio.numpy()
        
        # Создаем WAV файл в памяти
        byte_io = io.BytesIO()
        # Записываем WAV заголовок
        byte_io.write(b'RIFF')
        byte_io.write((36 + len(audio_np) * 2).to_bytes(4, 'little'))
        byte_io.write(b'WAVE')
        byte_io.write(b'fmt ')
        byte_io.write((16).to_bytes(4, 'little'))
        byte_io.write((1).to_bytes(2, 'little'))
        byte_io.write((1).to_bytes(2, 'little'))
        byte_io.write((sample_rate).to_bytes(4, 'little'))
        byte_io.write((sample_rate * 2).to_bytes(4, 'little'))
        byte_io.write((2).to_bytes(2, 'little'))
        byte_io.write((16).to_bytes(2, 'little'))
        byte_io.write(b'data')
        byte_io.write((len(audio_np) * 2).to_bytes(4, 'little'))
        
        # Записываем аудио данные
        audio_int16 = (audio_np * 32767).astype(np.int16)
        byte_io.write(audio_int16.tobytes())
        logger.info("WAV file created successfully")
        
        # Возвращаем в начало буфера
        byte_io.seek(0)
        
        return StreamingResponse(byte_io, media_type="audio/wav")
    except Exception as e:
        logger.error(f"Error in text_to_speech: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
