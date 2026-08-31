FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgles2 \
        libegl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --timeout 300 --retries 10 \
        torch==2.1.2+cpu \
        torchvision==0.16.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --timeout 300 --retries 10 -r requirements.txt

COPY . .

EXPOSE 8003

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
