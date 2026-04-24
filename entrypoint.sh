#!/usr/bin/env bash
#
#
set -e

# Do NOT set AWS region, set the DATASET_ENDPOINT_URL instead. See https://docs.aws.amazon.com/general/latest/gr/s3.html
DATASET_BUCKET=${DATASET_BUCKET:-"mmim"}
DATASET_IMAGES_BUCKET=${DATASET_IMAGES_BUCKET:-"mmim"}
DATASET_IMAGES_EXTENSION=${DATASET_IMAGES_EXTENSION:-dcm}
# Set DATASET_ENDPOINT_URL
DATASET_TRAINING_KEY=${DATASET_TRAINING_KEY:-"ds_train.csv"}
DATASET_VALIDATION_KEY=${DATASET_VALIDATION_KEY:-"ds_val.csv"}
DATASET_IMAGES_DIR=${DATASET_IMAGES_DIR:-"mimic-cxr-jpg"}
DATASET_TRAINING_FILENAME=${DATASET_TRAINING_FILENAME:-$(basename "$DATASET_TRAINING_KEY")}
DATASET_VALIDATION_FILENAME=${DATASET_VALIDATION_FILENAME:-$(basename "$DATASET_VALIDATION_KEY")}
export DATASET_LOCAL_DIR=${DATASET_LOCAL_DIR:-/app/dataset}
export BASE_DIR=${BASE_DIR:-/app}

# Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY

# if train and validation datasets exist no download
export TRAINING_DATASET_FILE=${TRAINING_DATASET_FILE:-"${DATASET_LOCAL_DIR}/${DATASET_TRAINING_FILENAME}"}
export VALIDATION_DATASET_FILE=${VALIDATION_DATASET_FILE:-"${DATASET_LOCAL_DIR}/${DATASET_VALIDATION_FILENAME}"}
export DATASET_IMAGES_BASEDIR=${DATASET_IMAGES_BASEDIR:-"${DATASET_LOCAL_DIR}/${DATASET_IMAGES_DIR}"}

echo "Creating datasets directory..."
train_ds_dir=$(dirname "$TRAINING_DATASET_FILE"); { [[ ! -d "$train_ds_dir" ]] && [[ ! -L "$train_ds_dir" ]] } && mkdir -p $(dirname "$TRAINING_DATASET_FILE")
val_ds_dir=$(dirname "$VALIDATION_DATASET_FILE"); { [[ ! -d "$val_ds_dir" ]] && [[ ! -L "$val_ds_dir" ]] } && mkdir -p $(dirname "$VALIDATION_DATASET_FILE")
echo "Creating images directory ${DATASET_IMAGES_BASEDIR}..."
{ [[ ! -d "$DATASET_IMAGES_BASEDIR" ]] && [[ ! -L "$DATASET_IMAGES_BASEDIR" ]] } && mkdir -p $(dirname "$DATASET_IMAGES_BASEDIR")

[[ ! -e "$TRAINING_DATASET_FILE" ]] \
  && echo "Downloading TRAINING dataset file..." \
  && aws s3 --endpoint-url "$DATASET_ENDPOINT_URL" cp s3://${DATASET_BUCKET}/${DATASET_TRAINING_KEY} ${TRAINING_DATASET_FILE}

[[ ! -e "$VALIDATION_DATASET_FILE" ]] \
  && echo "Downloading VALIDATION dataset file..." \
  && aws s3 --endpoint-url "$DATASET_ENDPOINT_URL" cp s3://${DATASET_BUCKET}/${DATASET_VALIDATION_KEY} ${VALIDATION_DATASET_FILE}

if [ -z "$(ls -A "$DATASET_IMAGES_BASEDIR" 2>/dev/null)" ]; then
  echo "Downloading dataset images, this may take a while..."
  aws s3 --endpoint-url "$DATASET_ENDPOINT_URL" sync s3://${DATASET_BUCKET}/${DATASET_IMAGES_DIR} ${DATASET_IMAGES_BASEDIR}
fi

echo "<=== Ready to train! ===>"
if ! cd "$BASE_DIR"; then
  echo "Unable to enter base directory. Exiting..." >&2
  exit 2
fi

# $BASE_DIR/.venv/bin already in PATH via Dockerfile!
exec uv run python trainer/main.py
