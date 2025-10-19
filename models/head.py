# models/head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionHead(nn.Module):
    """
    Takes (B, 512, Hf, Wf) features, outputs (B, S, S, 5).
    We do a 1x1 convolution to map 512-d features -> 5 outputs per cell.
    Output channels: [tx, ty, tw, th, obj]
    """
    def __init__(self, S: int):
        super().__init__()
        self.S = S
        self.conv1x1 = nn.Conv2d(512, 5, kernel_size=1, stride=1, padding=0)

    def forward(self, feats):
        # logits or probabilities? For simplicity we return probabilities (sigmoid),
        # consistent with your current loss setup using BCE on probs.
        out = self.conv1x1(feats)            # (B, 5, Hf, Wf)
        out = torch.sigmoid(out)             # (0..1)
        out = F.interpolate(out, size=(self.S, self.S), mode="bilinear", align_corners=False)
        out = out.permute(0, 2, 3, 1).contiguous()  # (B, S, S, 5)
        return out