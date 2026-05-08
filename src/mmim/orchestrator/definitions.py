from pathlib import Path

from dagster import definitions, load_from_defs_folder


@definitions
def definitions():
    return load_from_defs_folder(path_within_project=Path("defs"))
