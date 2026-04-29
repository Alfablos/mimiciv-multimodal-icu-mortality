from trainer.utils import find_paths


def test_find_nonexistent_paths():
    assert find_paths(
        ["tests/trainer/unit/nono.py", "tests/trainer/unit/test_data.py"]
    ) == ["tests/trainer/unit/nono.py"]
