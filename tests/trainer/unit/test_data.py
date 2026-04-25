from pytest import raises
import torch


from data import init_test_ds, current_features


def test_ds_has_right_features():
    test_ds = init_test_ds()
    assert test_ds.features == current_features
    for i in range(len(current_features)):
        assert test_ds.features[i] == current_features[i]


def test_ds_allowed_image_extensions_are_ok():
    for allowed_ext in ["jpg", ".jpg", "dcm", ".dcm", "dicom", ".dicom"]:
        _ = init_test_ds(images_extension=allowed_ext)


def test_ds_wrong_image_extensions_are_rejected():
    with raises(ValueError, match="Extension .+ is not supported."):
        init_test_ds(images_extension="unallowed")
        init_test_ds(images_extension=".unallowed")
        init_test_ds(images_extension="#? unallowed")


def test_ds_returns_images_correctly():
    ds = init_test_ds()
    paths = [
        "./tests/trainer/unit/images/p11/p11111111/s3030303/0.jpg",
        "./tests/trainer/unit/images/p22/p22222222/s8080808/1.jpg",
    ]

    for i, path in enumerate(paths):
        img, example, label = ds[i]
        assert example.shape.numel() == 34, "Wrong shape for training example"
        assert label.shape.numel() == 1, "Wrong shape for label"
        assert img.shape == torch.Size([3, 512, 512]), (
            f"Wrong shape for image: {img.shape}. Should be [3, 512, 512]"
        )
        assert ds.image_paths[i] == path
