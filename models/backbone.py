# models/backbone.py
import torch
import torch.nn as nn

def conv_block(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )

class Backbone(nn.Module):
    """
    Input:  (B, 3, 416, 416)
    Output: (B, 512, 13, 13)   # after downsampling with stride-2 convs
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(3,   16, k=3, s=2, p=1),  # 416 -> 208
            conv_block(16,  32, k=3, s=1, p=1),  # 208 -> 208
            conv_block(32,  64, k=3, s=2, p=1),  # 208 -> 104
            conv_block(64,  64, k=3, s=1, p=1),  # 104 -> 104
            conv_block(64, 128, k=3, s=2, p=1),  # 104 -> 52
            conv_block(128,128, k=3, s=1, p=1),  # 52  -> 52
            conv_block(128,256, k=3, s=2, p=1),  # 52  -> 26
            conv_block(256,256, k=3, s=1, p=1),  # 26  -> 26
            conv_block(256,512, k=3, s=2, p=1),  # 26  -> 13
            conv_block(512,512, k=3, s=1, p=1),  # 13  -> 13
        )

    def forward(self, x):
        return self.net(x)


class Backbone8x8(nn.Module):
    """
    Input:  (B, 3, 512, 512)
    Output: (B, 512, 8, 8) using high-capacity 512-channel blocks.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(3,   512, k=7, s=2, p=3),  # 512 -> 256
            conv_block(512, 512, k=3, s=2, p=1),  # 256 -> 128
            conv_block(512, 512, k=3, s=2, p=1),  # 128 -> 64
            conv_block(512, 512, k=3, s=2, p=1),  # 64  -> 32
            conv_block(512, 512, k=3, s=2, p=1),  # 32  -> 16
            conv_block(512, 512, k=3, s=2, p=1),  # 16  -> 8
            conv_block(512, 512, k=3, s=1, p=1),  # 8   -> 8
        )

    def forward(self, x):
        return self.net(x)


BACKBONE_REGISTRY = {
    "13x13": {"cls": Backbone, "grid": 13, "img_size": 416},
    "8x8": {"cls": Backbone8x8, "grid": 8, "img_size": 512},
}


def _get_backbone_entry(name: str):
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Available: {list(BACKBONE_REGISTRY.keys())}")
    return BACKBONE_REGISTRY[name]


def build_backbone(name: str) -> nn.Module:
    entry = _get_backbone_entry(name)
    return entry["cls"]()


def backbone_output_grid(name: str) -> int:
    entry = _get_backbone_entry(name)
    return entry["grid"]


def backbone_input_resolution(name: str) -> int:
    entry = _get_backbone_entry(name)
    return entry["img_size"]
