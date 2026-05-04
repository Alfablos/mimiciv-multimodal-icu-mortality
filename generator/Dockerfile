FROM astral/uv:python3.13-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_CACHE_DIR=/tmp/.uv-cache \
    PATH="/app/.venv/bin:$PATH"

RUN groupadd --system generator --gid 1000 \
    && useradd --system \
      --uid 1000 \
      --gid 1000 \
      --create-home \
      --shell /usr/sbin/nologin \
      generator \
    && mkdir -p /app /tmp/.uv-cache \
    && chown -R generator:generator /app /tmp/.uv-cache

WORKDIR /app

USER generator

COPY --chown=generator:generator ./pyproject.toml ./uv.lock ./

RUN uv sync --frozen --no-install-project

COPY --chown=generator:generator ./generator ./generator

RUN uv sync --frozen

ARG GIT_SHA
ARG GIT_REF

ENV GIT_SHA=${GIT_SHA} \
    GIT_REF=${GIT_REF}

LABEL org.opencontainers.image.revision=$GIT_SHA
LABEL org.opencontainers.image.ref.name=$GIT_REF
LABEL org.opencontainers.image.title=mmim-generator

ENTRYPOINT ["uv", "run", "python", "-m", "generator.main", "build"]
