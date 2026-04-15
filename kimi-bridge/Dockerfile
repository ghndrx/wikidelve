# Base: Microsoft's Playwright image ships Chromium + all the system libs
# pre-installed. We add Python+uv+kimi-cli on top. Heavier image (~1.5GB)
# but eliminates a whole class of "missing libnss" debugging.
FROM mcr.microsoft.com/playwright:v1.49.0-jammy

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/root/.local/bin:/usr/local/bin:$PATH

# Node is already in the Playwright image. Add Python 3 + curl for uv.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates curl git python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://astral.sh/uv/install.sh | sh \
 && /root/.local/bin/uv tool install --python 3.13 kimi-cli \
 && /root/.local/bin/uv tool update-shell \
 && kimi --version || true

WORKDIR /app
COPY package.json ./
RUN npm install --omit=dev

COPY server.js tools.js ./

ENV HOME=/root \
    PORT=5555 \
    NODE_ENV=production

EXPOSE 5555
CMD ["node", "server.js"]
