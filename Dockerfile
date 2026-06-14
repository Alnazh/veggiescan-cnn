# ================================================================
#  VeggieScan - Dockerfile untuk Hugging Face Spaces
#  Base: Python 3.11 slim (sesuai runtime.txt)
#  Port: 7860 (wajib di Hugging Face Spaces)
# ================================================================

FROM python:3.11-slim

# --- Metadata (opsional tapi baik untuk dokumentasi) ---
LABEL maintainer="VeggieScan"
LABEL description="Klasifikasi Sayuran CNN MobileNetV2 - Hugging Face Spaces"

# --- Hindari prompt interaktif saat apt install ---
ENV DEBIAN_FRONTEND=noninteractive

# --- Install system dependencies untuk OpenCV headless ---
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Set working directory ---
WORKDIR /app

# --- Copy requirements dan install duluan (layer cache) ---
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Copy seluruh source aplikasi ---
COPY . .

# --- Buat folder uploads agar Flask tidak error saat runtime ---
RUN mkdir -p static/img/uploads

# --- Hugging Face Spaces wajib pakai port 7860 ---
EXPOSE 7860

# --- Jalankan dengan gunicorn, 1 worker karena model besar ---
CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "1", \
     "--timeout", "120", \
     "--preload"]
