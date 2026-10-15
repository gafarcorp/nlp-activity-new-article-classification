FROM python:3.11-slim

WORKDIR /app
# WITH_BERT=1 installs the CPU build of PyTorch + transformers (large image). Use 0 to serve only
# Logistic Regression and Random Forest.
ARG WITH_BERT=1

COPY requirements.txt requirements-bert.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$WITH_BERT" = "1" ]; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
      && pip install --no-cache-dir -r requirements-bert.txt; \
    fi

# NLTK data is downloaded at build time so the container works offline afterwards
RUN python -c "import nltk; [nltk.download(r) for r in ('stopwords', 'wordnet', 'omw-1.4')]"

COPY backend ./backend
COPY frontend ./frontend

# trained models are mounted at run time (see docker-compose.yml)
ENV NEWSSORT_MODELS_DIR=/models
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
