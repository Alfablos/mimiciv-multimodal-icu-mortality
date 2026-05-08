from pytest import raises
import torch
import pandas as pd


from tests.unit.data import current_features, data, expected_image_paths, init_test_ds


def test_ds_has_right_features(tmp_path):
    test_ds = init_test_ds(tmp_path)
    assert test_ds.features == current_features
    for i in range(len(current_features)):
        assert test_ds.features[i] == current_features[i]


def test_ds_allowed_image_extensions_are_ok(tmp_path):
    empty_df = pd.DataFrame(data).iloc[0:0]
    for allowed_ext in ["jpg", ".jpg", "dcm", ".dcm", "dicom", ".dicom"]:
        _ = init_test_ds(tmp_path, images_extension=allowed_ext, df=empty_df)


def test_ds_wrong_image_extensions_are_rejected(tmp_path):
    with raises(ValueError, match="Extension .+ is not supported."):
        init_test_ds(tmp_path, images_extension="unallowed")
    with raises(ValueError, match="Extension .+ is not supported."):
        init_test_ds(tmp_path, images_extension=".unallowed")
    with raises(ValueError, match="Extension .+ is not supported."):
        init_test_ds(tmp_path, images_extension="#? unallowed")


def test_ds_returns_images_correctly(tmp_path):
    ds = init_test_ds(tmp_path)
    paths = expected_image_paths(tmp_path / "workdir")

    for i, path in enumerate(paths):
        img, example, label = ds[i]
        assert example.shape.numel() == 34, "Wrong shape for training example"
        assert label.shape.numel() == 1, "Wrong shape for label"
        assert img.shape == torch.Size([3, 512, 512]), (
            f"Wrong shape for image: {img.shape}. Should be [3, 512, 512]"
        )
        assert ds.image_paths[i] == path
