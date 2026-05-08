import json
from typing import Any
from pathlib import Path

from .base import ReadOnlyStore, WriteOnlyStore, gen_path


class FilesystemReadOnlyStore(ReadOnlyStore):
    def __init__(self, dir: str, prefix: str | None, debug: bool = False):
        self.debug = debug
        self.dir = Path(dir)
        if not self.dir.exists():
            raise FileNotFoundError(
                f"Cannot open filesystem store at `{dir}`: no such directory."
            )
        if self.dir.exists() and self.dir.is_file():
            raise FileExistsError(f"Path {self.dir} exists and it's a file.")
        if prefix is None:
            self.prefix = ""
        else:
            self.prefix = prefix

        if self.debug:
            print(
                f"Initialized filesystem read-only store in directory `{self.dir}` with default prefix `{self.prefix}`"
            )

    def _checks(self, fname: Path):
        if not fname.exists():
            raise FileNotFoundError(f"File {fname} does not exist.")
        if fname.is_dir():
            raise IsADirectoryError(f"Path {fname} is a directory.")

    def backend(self) -> str:
        return "filesystem"

    def read_text(self, path: str, with_prefix: bool = True) -> str:
        fname = gen_path(
            base=self.dir, suffix=path, prefix=self.prefix if with_prefix else None
        )
        if self.debug:
            print(
                f"Filesystem read-only store received text read request: path='{path}' prefix='{self.prefix if with_prefix else 'None'}'\nresulting_filename='{fname}'"
            )
        self._checks(fname)
        return fname.read_text()

    def read_bytes(self, path: str, with_prefix: bool = True) -> bytes:
        fname = gen_path(
            base=self.dir, suffix=path, prefix=self.prefix if with_prefix else None
        )
        if self.debug:
            print(
                f"Filesystem read-only store received bytes read request: path='{path}' prefix='{self.prefix if with_prefix else 'None'}'\nresulting_filename='{fname}'"
            )
        self._checks(fname)
        return fname.read_bytes()

    def set_prefix(self, prefix: str):
        if self.debug:
            print(
                f"Filesystem read-only store, prefix changed: {self.prefix} => {prefix}"
            )
        self.prefix = prefix


class FilesystemWriteOnlyStore(WriteOnlyStore):
    def __init__(self, dir: str, prefix: str | None, debug: bool = False):
        self.debug = debug
        self.dir = Path(dir)
        if self.dir.exists() and self.dir.is_file():
            raise FileExistsError(f"Path `{dir}` already exists and it's a file.")
        self.dir.mkdir(parents=True, exist_ok=True)

        if prefix is None:
            self.prefix = ""
        else:
            self.prefix = prefix

        if self.debug:
            print(
                f"Initialized filesystem write-only store in directory `{self.dir}` with default prefix `{self.prefix}`"
            )

    def write_text(
        self, path: str, data: str, overwrite: bool = False, with_prefix: bool = True
    ) -> None:
        fname = gen_path(
            base=self.dir, suffix=path, prefix=self.prefix if with_prefix else None
        )
        if self.debug:
            print(
                f"Filesystem write-only store received text write request: path='{path}' prefix='{self.prefix if with_prefix else 'None'}'\nresulting_filename='{fname}', overwrite='{overwrite}'"
            )
        if fname.exists() and not overwrite:
            raise FileExistsError(
                f"Path {fname} already exists in directory {self.dir}"
            )
        parent = fname.parent
        if not parent.exists():
            parent.mkdir(parents=True)
        fname.write_text(data)

    def write_bytes(
        self, path: str, data: bytes, overwrite: bool = False, with_prefix: bool = True
    ) -> None:
        fname = gen_path(
            base=self.dir, suffix=path, prefix=self.prefix if with_prefix else None
        )
        if self.debug:
            print(
                f"Filesystem write-only store received bytes write request: path='{path}' prefix='{self.prefix if with_prefix else 'None'}'\nresulting_filename='{fname}', overwrite='{overwrite}'"
            )
        if fname.exists() and not overwrite:
            raise FileExistsError(
                f"Path {fname} already exists in directory {self.dir}"
            )
        parent = fname.parent
        if not parent.exists():
            parent.mkdir(parents=True)
        fname.write_bytes(data)

    def write_file(
        self,
        local_path: str,
        remote_path: str,
        overwrite: bool = False,
        with_prefix: bool = True,
    ) -> None:
        fname = gen_path(
            base=self.dir,
            suffix=remote_path,
            prefix=self.prefix if with_prefix else None,
        )
        if self.debug:
            print(
                f"Filesystem write-only store received file write request: local_path='{local_path}' remote_path='{remote_path}' prefix='{self.prefix if with_prefix else 'None'}'\nresulting_filename='{fname}', overwrite='{overwrite}'"
            )
        if fname.exists() and not overwrite:
            raise FileExistsError(
                f"Path {fname} already exists in directory {self.dir}"
            )
        parent = fname.parent
        if not parent.exists():
            parent.mkdir(parents=True)
        with open(local_path, "rb") as source:
            with fname.open("wb") as target:
                target.write(source.read())

    def exists(self, path: str, with_prefix: bool = True) -> bool:
        fname = gen_path(
            base=self.dir, suffix=path, prefix=self.prefix if with_prefix else None
        )
        return fname.exists()

    def set_prefix(self, prefix: str):
        if self.debug:
            print(
                f"Filesystem write-only store, prefix changed: {self.prefix} => {prefix}"
            )
        self.prefix = prefix

    def commit(self, message: str, metadata: dict[str, Any] = {}) -> str:
        info = f"""
        {message}
        {json.dumps(metadata, indent=2)}
        """
        self.write_text(path="./info.txt", data=info, overwrite=True, with_prefix=False)
        return info
