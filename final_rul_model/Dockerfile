FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY app /app/app
COPY config /app/config
COPY model /app/model
COPY models /app/models
COPY scripts /app/scripts

ENV THRESHOLDS_PATH=/app/config/thresholds.json
ENV MODEL_CONFIG_PATH=/app/config/model_config.json
ENV BELT_METADATA_PATH=/app/config/belts_metadata.json
ENV MODEL_DIR=/app/models/saved_models

CMD ["python", "app/standalone_runtime.py"]