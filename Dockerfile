FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY train.py main.py schemas.py ./
COPY Dataset/ ./Dataset/

# Train on build so the image ships with a Production model baked in.
# MLflow uses a local SQLite DB inside the container.
RUN python train.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
