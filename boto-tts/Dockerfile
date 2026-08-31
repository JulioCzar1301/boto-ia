FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Manaus
ENV COQUI_TOS_AGREED=1

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt

# Substitui o torch padrão por uma build com suporte a Blackwell (sm_120)
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install --pre --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# RUN pip install --no-cache-dir torchcodec --index-url https://download.pytorch.org/whl/nightly/cu128

COPY . .

RUN mkdir -p embeddings voices

EXPOSE 8002

# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]