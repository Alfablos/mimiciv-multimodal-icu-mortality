FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# 1. Add UV_CACHE_DIR and set the PATH early
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_CACHE_DIR=/tmp/.uv-cache \
    PATH="/app/.venv/bin:$PATH"

RUN groupadd --system trainer --gid 1000 \
    && useradd --system \
      --uid 1000 \
      --gid 1000 \
      --create-home \
      --shell /usr/sbin/nologin \
      trainer \
    && mkdir -p /app/dataset \
    && chown -R trainer:trainer /app

RUN apt update && apt install -y --no-install-recommends \
    curl \
    awscli \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

WORKDIR /app

USER trainer

COPY --chown=trainer:trainer ./uv.lock ./pyproject.toml ./

RUN uv sync --frozen --no-install-project

COPY --chown=trainer:trainer ./trainer ./trainer
RUN uv sync --frozen

# changes often and invalidates cache
USER root
COPY --chmod=755 ./entrypoint.sh /entrypoint.sh
USER trainer

ENTRYPOINT [ "/entrypoint.sh" ]
