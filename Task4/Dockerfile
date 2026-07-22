FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_HOST=http://ollama:11434 \
    OLLAMA_MODEL=qwen2.5:3b \
    INDEX_DIR=/quantum-forge-bot/faiss_index \
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    HF_HOME=/root/.cache/huggingface

# FAISS uses OpenMP on Debian-based Python images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /quantum-forge-bot

COPY requirements-docker.txt ./requirements-docker.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-docker.txt

# The image remains runnable even without development bind mounts.
COPY Task4 ./Task4
COPY faiss_index ./faiss_index

WORKDIR /quantum-forge-bot/Task4
CMD ["python", "rag_bot.py"]
