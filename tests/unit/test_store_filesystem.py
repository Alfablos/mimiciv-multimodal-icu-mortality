from pytest import raises

from mmim.store.filesystem import FilesystemReadOnlyStore, FilesystemWriteOnlyStore


def test_filesystem_store_respects_prefix_for_reads_and_exists(tmp_path):
    base = tmp_path / "store"
    prefixed = base / "dataset" / "v001"
    prefixed.mkdir(parents=True)
    (prefixed / "data.txt").write_text("hello")

    store = FilesystemReadOnlyStore(str(base), prefix="dataset/v001")

    assert store.read_text("data.txt") == "hello"

    writer = FilesystemWriteOnlyStore(str(base), prefix="dataset/v001")
    assert writer.exists("data.txt") is True
    assert writer.exists("missing.txt") is False


def test_filesystem_write_text_overwrite_semantics(tmp_path):
    store = FilesystemWriteOnlyStore(str(tmp_path), prefix="dataset")
    store.write_text("value.txt", "first")

    with raises(FileExistsError):
        store.write_text("value.txt", "second", overwrite=False)

    store.write_text("value.txt", "second", overwrite=True)

    assert (tmp_path / "dataset" / "value.txt").read_text() == "second"


def test_filesystem_write_bytes_overwrite_semantics(tmp_path):
    store = FilesystemWriteOnlyStore(str(tmp_path), prefix="dataset")
    store.write_bytes("value.bin", b"first")

    with raises(FileExistsError):
        store.write_bytes("value.bin", b"second", overwrite=False)

    store.write_bytes("value.bin", b"second", overwrite=True)

    assert (tmp_path / "dataset" / "value.bin").read_bytes() == b"second"


def test_filesystem_write_file_overwrite_semantics(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("first")

    store = FilesystemWriteOnlyStore(str(tmp_path / "store"), prefix="dataset")
    store.write_file(str(source), "copied.txt")

    source.write_text("second")
    with raises(FileExistsError):
        store.write_file(str(source), "copied.txt", overwrite=False)

    store.write_file(str(source), "copied.txt", overwrite=True)

    assert (tmp_path / "store" / "dataset" / "copied.txt").read_text() == "second"


def test_filesystem_commit_overwrites_info_file(tmp_path):
    store = FilesystemWriteOnlyStore(str(tmp_path), prefix="dataset")

    store.commit("first", {"a": "b"})
    store.commit("second", {"c": "d"})

    assert "second" in (tmp_path / "info.txt").read_text()
