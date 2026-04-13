FROM python:3.12-slim AS builder

WORKDIR /build

# Upgrade pip first — the stock 25.0.1 shipped with python:3.12-slim
# carries CVE-2025-8869 (symlink tar extraction) and CVE-2026-1703
# (wheel path traversal). Both are fixed in 26.0+.
RUN pip install --no-cache-dir --upgrade 'pip>=26.0' 'setuptools>=78'

COPY requirements.txt /build/requirements.txt

RUN pip install --no-cache-dir --prefix=/install -r /build/requirements.txt

FROM python:3.12-slim

WORKDIR /app

# Runtime pip is only here for ad-hoc operations; keep it patched too.
RUN pip install --no-cache-dir --upgrade 'pip>=26.0' 'setuptools>=78'

COPY --from=builder /install /usr/local

COPY app/ /app/app/
COPY templates/ /app/templates/

EXPOSE 8888

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8888", "--no-server-header", "--no-date-header"]
