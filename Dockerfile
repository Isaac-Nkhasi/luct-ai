# Phase 10 — container image for the LUCT Market price-estimation service.
#
# The base image matches the interpreter the model was trained with (Python 3.14)
# and api/requirements.txt pins the exact library versions, so the joblib pickle
# loads without a scikit-learn version mismatch.

FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so this layer is reused when only the source changes.
# Two separate COPYs: a multi-source COPY with a directory destination flattens
# both files to their basename, which would clobber one of them.
COPY requirements.txt ./
COPY api/requirements.txt ./api/
RUN pip install --no-cache-dir -r requirements.txt -r api/requirements.txt

# Editable install keeps src/ in place, so luct_ai.serving resolves <repo>/models.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .

# The service code and the trained model bundle it loads at startup.
COPY api ./api
COPY models ./models

EXPOSE 8000

# Render injects $PORT; fall back to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
