from os import cpu_count
import os

from .utils import int_from_env, float_from_env, bool_from_env

# Available environment variables:
# DATASET_LOCAL_DIR
# BASE_DIR
# TRAINING_DATASET_FILE
# VALIDATION_DATASET_FILE
# DATASET_STATS_FILE
# DATASET_IMAGES_BASEDIR
# MLFLOW_TRACKING_URI
# MLFLOW_TRACKING_USERNAME
# MLFLOW_TRACKING_PASSWORD
#
# Can also set:
# MMIM_BATCH_SIZE
# MMIM_EPOCHS
# MMIM_DROPOUT
# MMIM_TRAIN_LIMIT
# MMIM_DEBUG


# survivors / deaths in the training set.
# needed due to heavily imbalanced label
# This is a dataset property, do NOT modify unless you know what you're doing!
# Be VERY CAREFUL when modifying `train_limit` as it really can impact this value!
loss_pos_weight = float_from_env(
    "MMIM_LOSS_POS_WEIGHT", 5160 / 936
)  # positive weight is negatives / positives = (6096 - 936) / 936 !!

debug = os.getenv("MMIM_DEBUG", "false").lower() in ["true", "1", "yes"]

train_csv = os.getenv("TRAINING_DATASET_FILE", "./dataset/ds_train.csv")
val_csv = os.getenv("VALIDATION_DATASET_FILE", "./dataset/ds_val.csv")
dataset_stats_file = os.getenv("DATASET_STATS_FILE", "./dataset/stats.json")

image_base_dir = os.getenv(
    "DATASET_IMAGES_BASEDIR",
    "./dataset/mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.1.0/files",
)
image_extension = os.getenv("DATASET_IMAGES_EXTENSION", "dcm")

dataset_shuffle = bool_from_env("MMIM_DATASET_SHUFFLE", True)
default_num_workers = max(((cpu_count() or 1) // 2) - 2, 0)
num_workers = int_from_env("MMIM_NUM_WORKERS", default_num_workers)

model_selection_metric = "AUPRC"

hyperparameters = {
    "batch_size": int_from_env("MMIM_BATCH_SIZE", 32),
    "epochs": int_from_env("MMIM_EPOCHS", 10),
    "dropout": float_from_env("MMIM_DROPOUT", 0.3),
    "learning_rate": float_from_env("MMIM_LEARNING_RATE", 10e-4),
    "train_limit": float_from_env("MMIM_TRAIN_LIMIT", 1.0),
}
