from attr import dataclass


@dataclass
class TResult:
    start_time: str
    end_time: str
    train_status: str


def get_hyperparameters() -> dict[str, int | float | str]:
    pass


def get_dataset() -> str:
    pass


def get_trainer(
    dataset: str, hyperparameters: dict[str, int | float], working_directory: str | None
) -> TResult: ...
