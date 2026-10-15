FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HOME=/home/raven YOLO_CONFIG_DIR=/home/raven/.config/Ultralytics
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 libgomp1 ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && pip install -r requirements.txt
COPY configs ./configs
COPY src ./src
COPY migrations ./migrations
# Ultralytics downloads named pretrained detector weights on first use.
# Include the small project checkpoint; bulky local weights stay outside the image.
COPY data/models/confidence_fusion.pt ./data/models/confidence_fusion.pt
RUN useradd --create-home --uid 10001 raven && \
    mkdir -p data/objects data/models data/alerts "$YOLO_CONFIG_DIR" && \
    chown -R raven:raven /app/data /home/raven
USER raven
EXPOSE 8000
