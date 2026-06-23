from typing import Protocol, Any, Literal, Self, TypeVar, Callable, Generator


class Artifact(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def kind(self) -> Literal["image", "figure", "json", "text", "binary"]: ...
    @property
    def data(self) -> Any: ...
    @property
    def filename(self) -> str: ...


class Loggable(Protocol):
    @property
    def model(self) -> Any: ...

    @property
    def epoch(self) -> int: ...

    @property
    def metrics(self) -> dict[str, int | float]: ...

    @property
    def metadata(self) -> dict[str, int | float | str]: ...

    @property
    def val_loss(self) -> float: ...

    @property
    def train_loss(self) -> float: ...

    @property
    def train_start_time(self) -> str: ...

    @property
    def train_end_time(self) -> str: ...

    @property
    def selection_metric(self) -> str | None: ...

    @property
    def selection_metric_value(self) -> float | int | None: ...

    # This method returns a function that explains to the platform
    # how to generate artifacts
    # Only the platform knows if the model is going to be logged
    # or not and artifact should only be generated on logging
    # CAREFUL when capturing a MODEL! Techniques like GradCAM need
    # the model to be in `eval`: be sure to pass a deep clone or
    # do not use the model while gradcam is being performed
    # TODO: handle this better.
    def get_artifacts(self) -> Callable[[], list[Artifact]]: ...


# completed: completed, interrupted: user interrupt, failed: handled fatal exceptions
type TrainStatus = Literal["completed", "interrupted", "failed"]


class TrainingSummary(Protocol):
    @property
    def start_time(self) -> str: ...

    @property
    def end_time(self) -> str: ...

    @property
    def train_status(self) -> TrainStatus: ...


class Hyperparameters(Protocol):
    def to_json(self) -> str: ...

    @classmethod
    def from_json(cls, j: str) -> Self: ...


# D is a placeholder for the dataset "entrypoint"
# The platform will pass it to the trainer as-is
D = TypeVar("D")


class DatasetRef[D](Protocol):
    @property
    def manifest_uri(self) -> str: ...

    @property
    def root_uri(self) -> str: ...

    @property
    def inner(self) -> D: ...


## Expected from the client:
# The platform can expose the possibility to modify hyperparameters maybe via a UI
def get_hyperparameters() -> Hyperparameters: ...


# The platform owns the dataset but doesn't know how to use it: it only
# cares about being able to locate it and register its manifest
def get_dataset[D]() -> DatasetRef[D]: ...


# The platform can autonomously start training jobs
def get_trainer[D]() -> Callable[
    [DatasetRef[D], Hyperparameters], Generator[Loggable, bool, TrainingSummary]
]: ...


# Note: Generator[Loggable, bool, TrainingSummary] means:
# - The trainer has to yield a Loggable on each iteration
# - The platform SENDS True IF THE MODEL IS WORTH LOGGING
# - The return value is a TrainingSummary (TODO: really needed??)
