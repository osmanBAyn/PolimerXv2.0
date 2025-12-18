# 1. Temel Sistem
FROM python:3.10-slim

# 2. Gereksiz dosyaların önbelleğini temizleyerek Linux kütüphanelerini kur
RUN apt-get update && apt-get install -y \
    build-essential \
    libxrender1 \
    libxext6 \
    libsm6 \
    libx11-dev \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Çalışma klasörü
WORKDIR /app

# 4. ÖNCE PyTorch CPU Sürümünü Yükle (En büyük tasarruf burada!)
# --no-cache-dir: İndirilen kurulum dosyalarını saklama (yer kazandırır)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 5. Şimdi diğer gereksinimleri yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Dosyaları kopyala
COPY . .

# 7. Portu Aç ve Başlat
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
