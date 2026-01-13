FROM python:3.12-slim


# prevents python from writing pyc files to disc
# and ensures that Python output is sent straight to terminal (e.g. your container log)s)
# and avoids caching pip downloads
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# uvicorn src.app_textblob:app --host 0.0.0.0 --port 8000 --reload
# IMPORTANT: bind to 0.0.0.0, not localhost
CMD ["uvicorn", "src.app_textblob:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]