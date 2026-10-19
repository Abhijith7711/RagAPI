FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY sample_documents ./sample_documents

RUN mkdir -p /app/data/chroma /app/.cache/huggingface && chown -R appuser:appuser /app

USER appuser
ENV HOME=/home/appuser
ENV HF_HOME=/app/.cache/huggingface
ENV CHROMA_PERSIST_DIRECTORY=/app/data/chroma
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
