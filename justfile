

default:
  @ just --list

image:
  #!/usr/bin/env bash
  set -euo pipefail
  GIT_SHA="$(git rev-parse HEAD)"
  GIT_REF="$(git symbolic-ref --quiet --short HEAD)"
  IMAGE_TAG="${GIT_REF}-${GIT_SHA}"

  docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
