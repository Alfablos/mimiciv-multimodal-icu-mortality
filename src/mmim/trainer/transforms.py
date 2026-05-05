import torch
import torchvision.transforms.v2 as transformsV2


class PadToSquare(transformsV2.Transform):
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # incoming image shape should be (C, H, W)
        _, h, w = img.shape

        if h == w:
            return img

        max_dim = max(h, w)

        # calculating pad right instead of assuming it's equal
        # to pad left accounts for uneven sizes
        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top

        # fill=0 adds black padding
        return transformsV2.functional.pad(
            img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0
        )
