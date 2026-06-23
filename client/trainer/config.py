from os import cpu_count
import os

from pydantic import BaseModel

from utils import int_from_env, float_from_env, bool_from_env

# Available environment variables:
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
# loss_pos_weight = read from the dataset manifest

debug = bool_from_env("MMIM_TRAINER_DEBUG", False)


dataset_shuffle = bool_from_env("MMIM_TRAINER_DATASET_SHUFFLE", True)
default_num_workers = max(((cpu_count() or 1) // 2) - 2, 0)
num_workers = int_from_env("MMIM_TRAINER_NUM_WORKERS", default_num_workers)
working_directory = os.getenv("MMIM_TRAINER_WORKING_DIRECTORY", "./")
model_selection_metric = os.getenv("MMIM_TRAINER_MODEL_SELECTION_METRIC", "AUROC")

experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Multimodal ICU mortality")
experiment_family = experiment_name.lower().replace(" ", "_")


class Hyperparameters(BaseModel):
    batch_size: int = int_from_env("MMIM_TRAINER_BATCH_SIZE", 32)
    epochs: int = int_from_env("MMIM_TRAINER_EPOCHS", 10)
    dropout: float = float_from_env("MMIM_TRAINER_DROPOUT", 0.3)
    learning_rate: float = float_from_env("MMIM_TRAINER_LEARNING_RATE", 10e-4)
    train_limit: float = float_from_env("MMIM_TRAINER_TRAIN_LIMIT", 1.0)

    def summary(self) -> dict[str, int | float]:
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "train_limit": self.train_limit,
        }
