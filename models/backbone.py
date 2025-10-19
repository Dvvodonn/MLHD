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
    Output: (B, 512, 26, 26)   # after downsampling with stride-2 convs
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
            conv_block(256,512, k=3, s=1, p=1),  # 26  -> 26
        )

    def forward(self, x):
        return self.net(x)