from typing import Any
from pathlib import Path
from abc import ABC, abstractmethod


def gen_path(base: Path, prefix: str | None, suffix: str) -> Path:
    if prefix is None:
        return base / suffix
    else:
        return base / prefix / suffix


class ReadOnlyStore(ABC):
    @abstractmethod
    def read_text(self, path: str, with_prefix: bool) -> str:
        raise NotImplementedError

    @abstractmethod
    def read_bytes(self, path: str, with_prefix: bool) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def set_prefix(self, prefix: str) -> None:
        raise NotImplementedError


class WriteOnlyStore(ABC):
    @abstractmethod
    def write_text(
        self, path: str, data: str, overwrite: bool, with_prefix: bool
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def write_bytes(
        self, path: str, data: bytes, overwrite: bool, with_prefix: bool
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def write_file(
        self, local_path: str, remote_path: str, overwrite: bool, with_prefix: bool
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def exists(self, path: str, with_prefix: bool) -> bool:
        raise NotImplementedError

    @abstractmethod
    def commit(self, message: str, metadata: dict[str, Any] = {}) -> str:
        raise NotImplementedError

    @abstractmethod
    def set_prefix(self, prefix: str) -> None:
        raise NotImplementedError
