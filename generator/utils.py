from pathlib import Path


def find_paths(paths: list[str]) -> list[str]:
    return [p for p in paths if not Path(p).exists()]
