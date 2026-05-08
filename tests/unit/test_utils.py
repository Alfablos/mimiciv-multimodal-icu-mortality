from mmim.trainer.utils import find_paths


def test_find_nonexistent_paths():
    assert find_paths(["tests/unit/nono.py", "tests/unit/test_data.py"]) == [
        "tests/unit/nono.py"
    ]
