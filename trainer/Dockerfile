FROM astral/uv:python3.13-trixie

WORKDIR /app

ARG GIT_SHA
ARG GIT_REF

ENV GIT_SHA=${GIT_SHA}
ENV GIT_REF=${GIT_REF}
ENV UV_CACHE_DIR=/tmp/.uv-cache
ENV PATH="/app/.venv/bin:${PATH}"

LABEL org.opencontainers.image.revision=${GIT_SHA} \
      org.opencontainers.image.ref.name=${GIT_REF} \
      org.opencontainers.image.title=mmim-trainer

RUN useradd --create-home --shell /usr/sbin/nologin trainer \
    && mkdir -p /app /tmp/.uv-cache

COPY ./pyproject.toml ./pyproject.toml
COPY ./uv.lock ./uv.lock
COPY ./trainer ./src

RUN uv sync --frozen --no-dev

COPY ./entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

RUN chown -R trainer:trainer /app /tmp/.uv-cache

USER trainer

CMD ["python", "-m", "trainer.main", "--help"]
