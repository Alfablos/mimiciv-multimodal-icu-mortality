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
    def backend(self) -> str:
        raise NotImplementedError

    # TODO: support passing a lambda to transform from the reader
    # + returning ContextManager[TextIO] and ContextManager[BinaryIO] in open_text() and open_binary()
    # @staticmethod
    # open_text_and_then(self, path: str, op: Callable[[TextIO], T], with_prefix: bool = True)
    # @staticmethod
    # open_binary_and_then(self, path: str, op: Callable[[BinaryIO], T], with_prefix: bool = True)
    @abstractmethod
    def read_text(self, path: str, with_prefix: bool = True) -> str:
        raise NotImplementedError

    @abstractmethod
    def read_bytes(self, path: str, with_prefix: bool = True) -> bytes:
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
