#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="/app/dataset"
IMAGE_TREE="${DATASET_IMAGES_BASEDIR:-/app/dataset/mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.1.0/files}"

FILES=(
    "${BUNDLE_DIR}/ds_train.csv"
    "${BUNDLE_DIR}/ds_val.csv"
    "${BUNDLE_DIR}/ds_test.csv"
    "${BUNDLE_DIR}/stats.json"
    "${BUNDLE_DIR}/manifest.json"
    "${BUNDLE_DIR}/schema.json"
)

MISSING=()
for f in "${FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        MISSING+=("$f")
    fi
done
if [[ ! -d "$IMAGE_TREE" ]]; then
    MISSING+=("${IMAGE_TREE} (image tree directory)")
fi

if [[ ${#MISSING[@]} -gt 0 ]]; then
    printf 'ERROR: Missing required dataset artifacts:\n' >&2
    for m in "${MISSING[@]}"; do
        printf '  - %s\n' "$m" >&2
    done
    exit 1
fi

exec python -m trainer.main train
