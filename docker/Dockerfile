# syntax=docker/dockerfile:1

# ---------- Stage 1: build the wheel ----------
FROM python:3.12-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir hatchling

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m hatchling build -t wheel


# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime

# Turkish data flows through the whole app; force UTF-8 everywhere.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

# Run as a non-root user.
RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app

COPY --from=builder /build/dist/*.whl /tmp/
# The compose stack talks to PostgreSQL and the UI service reuses this image.
RUN pip install --no-cache-dir /tmp/*.whl \
        "psycopg[binary]>=3.1" "streamlit>=1.35" "plotly>=5.22" \
    && rm -rf /tmp/*.whl

COPY app ./app

RUN mkdir -p /app/data && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "roommate_matcher.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
