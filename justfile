

default:
  @ just --list

image:
  #!/usr/bin/env bash
  set -euo pipefail
  GIT_SHA="$(git rev-parse HEAD)"
  GIT_REF="$(git symbolic-ref --quiet --short HEAD)"
  IMAGE_TAG="${GIT_REF}-${GIT_SHA}"

  docker build \
    --build-arg GIT_SHA="${GIT_SHA}" \
    --build-arg GIT_REF="${GIT_REF}" \
    --label org.opencontainers.image.revision="${GIT_SHA}" \
    --label org.opencontainers.image.ref.name="${GIT_REF}" \
    -t ${IMAGE_NAME}:${IMAGE_TAG} .

train-container:
  #!/usr/bin/env bash
  set -euo pipefail
  GIT_SHA="$(git rev-parse HEAD)"
  GIT_REF="$(git symbolic-ref --quiet --short HEAD)"
  docker compose build \
    --build-arg GIT_SHA \
    --build-arg GIT_REF &&
    docker compose up
