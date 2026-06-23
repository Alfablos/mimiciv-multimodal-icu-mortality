from typing import Protocol


class DatasetBuildOutput(Protocol):
    def output_dir(self) -> str | None: ...
    def lakefs_ref(self) -> str | None: ...


class DatasetBuilder(Protocol):
    def build(self) -> DatasetBuildOutput: ...
