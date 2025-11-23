# models/yolo_like.py
import torch.nn as nn
from .backbone import build_backbone
from .head import DetectionHead

class Model(nn.Module):
    def __init__(self, S: int = 13, backbone_name: str = "13x13"):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = build_backbone(backbone_name)
        self.head = DetectionHead(S=S)

    def forward(self, x):
        feats = self.backbone(x)  # <-- CONVOLUTIONS HAPPEN HERE
        out   = self.head(feats)  # <-- 1x1 conv + resize to (S,S,5)
        return out
