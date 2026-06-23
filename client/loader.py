from typing import Generator, Callable

from config import Hyperparameters
from train import LoggedModel, TrainLoopResult


def get_hyperparameters() -> dict[str, int | float | str]:
    return Hyperparameters().model_dump()


def get_dataset() -> str:
    pass


def get_trainer(
    dataset: str, hyperparameters: dict[str, int | float], working_directory: str | None
) -> Callable[[str], Generator[LoggedModel, bool, TrainLoopResult]]: ...
