from typing import Any, cast
from pathlib import Path

from lakefs import Repository, Branch
from lakefs.reference import Reference


from .base import ReadOnlyStore, WriteOnlyStore, gen_path


class LakeFSReadOnlyStore(ReadOnlyStore):
    def __init__(self, ref: Reference, prefix: str | None, debug: bool = False):
        self.debug = debug
        self.repository = ref.repo_id
        self.ref = ref
        if prefix is None:
            self.prefix = ""
        else:
            self.prefix = prefix

        if self.debug:
            print(
                f"Initialized LakeFSReadOnlyStore in ref `{self.ref}` (repo `{self.repository}`) with default prefix `{self.prefix}`"
            )

    def backend(self) -> str:
        return "lakefs"

    def read_text(self, path: str, with_prefix: bool = True) -> str:
        fname = gen_path(
            base=Path("."), suffix=path, prefix=self.prefix if with_prefix else None
        )
        if self.debug:
            print(
                f"LakeFSReadOnlyStore: received text read request: path='{path}' prefix='{self.prefix if with_prefix else 'None'}'\nresulting_filename='{fname}'"
            )
        return cast(str, self.ref.object(f"{fname}").reader(mode="r").read())

    def read_bytes(self, path: str, with_prefix: bool = True) -> bytes:
        fname = gen_path(
            base=Path("."), suffix=path, prefix=self.prefix if with_prefix else None
        )
        if self.debug:
            print(
                f"LakeFSReadOnlyStore: received bytes read request: path='{path}' prefix='{self.prefix if with_prefix else 'None'}'\nresulting_filename='{fname}'"
            )
        return cast(bytes, self.ref.object(f"{fname}").reader(mode="rb").read())

    def exists(self, path: str, with_prefix: bool = True) -> bool:
        fname = gen_path(
            base=Path("."), suffix=path, prefix=self.prefix if with_prefix else None
        )
        return self.ref.object(f"{fname}").exists()

    def set_prefix(self, prefix: str):
        if self.debug:
            print(f"LakeFSReadOnlyStore: prefix changed {self.prefix} => {prefix}")
        self.prefix = prefix

    def get_ref(self):
        return self.ref


class LakeFSWriteOnlyStore(WriteOnlyStore):
    def __init__(
        self,
        repository: Repository,
        base_branch: str | None,
        branch: str,
        prefix: str | None,
        debug: bool = False,
    ):
        self.debug = debug
        base_branch = "master" if base_branch is None else base_branch
        self.repo = repository
        self.branch = repository.branch(branch).create(
            source_reference=base_branch, exist_ok=True
        )
        if prefix is None:
            self.prefix = ""
        else:
            self.prefix = prefix
        if self.debug:
            print(
                f"Initialized LakeFSWriteOnlyStore in branch `{self.branch}` (repo `{self.repo}`) with default prefix `{self.prefix}`"
            )

    def write_text(
        self, path: str, data: str, overwrite: bool = False, with_prefix: bool = True
    ) -> None:
        fname = gen_path(
            base=Path("."), suffix=path, prefix=self.prefix if with_prefix else None
        )
        if self.debug:
            print(
                f"LakeFSWriteOnlyStore: received text write request: path='{path}' prefix='{self.prefix if with_prefix else 'None'}'\nresulting_filename='{fname}', overwrite='{overwrite}'"
            )
        write_mode = "w" if overwrite else "x"
        obj = self.branch.object(f"{fname}")
        if obj.exists() and not overwrite:
            raise FileExistsError(
                f"LakeFSWriteOnlyStore: path {fname} already exists in branch {self.branch.id} of repo {self.repo.id}."
            )
        with obj.writer(mode=write_mode) as w:
            w.write(data)

    def write_bytes(
        self, path: str, data: bytes, overwrite: bool = False, with_prefix: bool = True
    ) -> None:
        fname = gen_path(
            base=Path("."), suffix=path, prefix=self.prefix if with_prefix else None
        )
        if self.debug:
            print(
                f"LakeFSWriteOnlyStore: received bytes write request: path='{path}' prefix='{self.prefix if with_prefix else 'None'}'\nresulting_filename='{fname}', overwrite='{overwrite}'"
            )
        write_mode = "wb" if overwrite else "xb"
        obj = self.branch.object(f"{fname}")
        if obj.exists() and not overwrite:
            raise FileExistsError(
                f"LakeFSWriteOnlyStore: Path {fname} already exists in branch {self.branch.id} of repo {self.repo.id}."
            )
        if self.debug:
            print(
                f"LakeFSWriteOnlyStore: writing {fname} to branch {self.branch.id} of repo {self.repo.id}"
            )
        with obj.writer(mode=write_mode) as w:
            w.write(data)

    def exists(self, path, with_prefix: bool = True):
        fname = gen_path(
            base=Path("."), suffix=path, prefix=self.prefix if with_prefix else None
        )
        return self.branch.object(f"{fname}").exists()

    def commit(self, message: str, metadata: dict[str, Any] = {}) -> str:
        return self.branch.commit(message=message, metadata=metadata).id

    def set_prefix(self, prefix: str):
        if self.debug:
            print(f"LakeFSWriteOnlyStore: prefix changed {self.prefix} => {prefix}")
        self.prefix = prefix

    def write_file(
        self,
        local_path: str,
        remote_path: str,
        overwrite: bool = False,
        with_prefix: bool = True,
    ) -> None:
        fname = gen_path(
            base=Path("."),
            suffix=remote_path,
            prefix=self.prefix if with_prefix else None,
        )
        obj = self.branch.object(f"{fname}")
        exists = obj.exists()
        if exists and not overwrite:
            raise FileExistsError(f"Path {fname} already exists.")

        if self.debug:
            print(
                f"LakeFSWriteOnlyStore: writing {fname} to branch {self.branch.id} of repo {self.repo.id}"
            )
        write_mode = "wb" if overwrite else "xb"
        with open(local_path, "rb") as local:
            with obj.writer(mode=write_mode) as remote:
                remote.write(local.read())

    def get_branch(self) -> Branch:
        return self.branch
