FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for FAISS and PyMuPDF
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create necessary directories and set permissions
RUN mkdir -p data/uploads data/vector_store && chmod -R 777 data

# Set PYTHONPATH to ensure 'app' module is found
ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
