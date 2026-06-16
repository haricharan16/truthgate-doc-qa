FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create data and log directories
RUN mkdir -p data/chroma data/corpus logs

# Expose API port
EXPOSE 8000

# Default: start API (assumes data/ already populated via volume or prior ingest)
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
