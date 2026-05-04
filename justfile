default:
  @ just --list

generator *args:
  @uv --project generator run python -m generator.main {{args}}

trainer *args:
  @uv --project trainer run python -m trainer.main {{args}}

test:
  @uv run --group dev pytest -q

generator-test:
  @uv run --project generator --group dev pytest -q generator/tests

trainer-test:
  @uv run --project trainer --group dev pytest -q trainer/tests

generator-image *args:
  #!/usr/bin/env bash
  set -euo pipefail
  GIT_SHA="$(git rev-parse HEAD)"
  GIT_REF="$(git symbolic-ref --short HEAD)"
  IMAGE_TAG="${GIT_REF}-${GIT_SHA}"
  docker build \
    -f generator/Dockerfile \
    --build-arg GIT_SHA="${GIT_SHA}" \
    --build-arg GIT_REF="${GIT_REF}" \
    -t mmim-generator:${IMAGE_TAG} \
    {{args}} \
    .

trainer-image *args:
  #!/usr/bin/env bash
  set -euo pipefail
  GIT_SHA="$(git rev-parse HEAD)"
  GIT_REF="$(git symbolic-ref --short HEAD)"
  IMAGE_TAG="${GIT_REF}-${GIT_SHA}"
  docker build \
    -f trainer/Dockerfile \
    --build-arg GIT_SHA="${GIT_SHA}" \
    --build-arg GIT_REF="${GIT_REF}" \
    -t mmim-trainer:${IMAGE_TAG} \
    {{args}} \
    .

generator-compose-up:
  #!/usr/bin/env bash
  set -euo pipefail
  docker compose -f generator/compose.yml up --build

trainer-compose-up:
  #!/usr/bin/env bash
  set -euo pipefail
  docker compose -f trainer/compose.yml up --build

# Prove generator↔trainer boundaries are intact
check-boundaries:
  @uv run --project generator --group dev pytest -q generator/tests/unit/test_generator_project_boundary.py
  @uv run --project trainer --group dev pytest -q trainer/tests/unit/test_trainer_project_boundary.py trainer/tests/unit/test_trainer_artifact_boundary.py trainer/tests/unit/test_dataset_contract.py trainer/tests/unit/test_trainer_entrypoint.py trainer/tests/unit/test_trainer_compose.py trainer/tests/unit/test_trainer_container_image.py
