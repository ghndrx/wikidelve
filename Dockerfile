FROM python:3.12-slim AS builder

WORKDIR /build

# Build-time deps for xhtml2pdf's transitive deps (pycairo + pillow).
# Confined to the builder stage; runtime image only carries the
# matching shared libraries below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        pkg-config \
        meson \
        ninja-build \
        libcairo2-dev \
        libjpeg-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first — the stock 25.0.1 shipped with python:3.12-slim
# carries CVE-2025-8869 (symlink tar extraction) and CVE-2026-1703
# (wheel path traversal). Both are fixed in 26.0+.
RUN pip install --no-cache-dir --upgrade 'pip>=26.0' 'setuptools>=78'

COPY requirements.txt /build/requirements.txt

RUN pip install --no-cache-dir --prefix=/install -r /build/requirements.txt

FROM python:3.12-slim

WORKDIR /app

# Runtime shared libs needed by the wheels above (~25MB).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcairo2 \
        libjpeg62-turbo \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Runtime pip is only here for ad-hoc operations; keep it patched too.
RUN pip install --no-cache-dir --upgrade 'pip>=26.0' 'setuptools>=78'

COPY --from=builder /install /usr/local

COPY app/ /app/app/
COPY templates/ /app/templates/

EXPOSE 8888

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8888", "--no-server-header", "--no-date-header"]
