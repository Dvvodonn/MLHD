import torch
from typing import List, Tuple


def load_yolo_labels(label_path: str) -> List[Tuple[float, float, float, float]]:
    """
    Parse YOLO format label file (class_id cx cy w h per line).
    Returns list of (cx, cy, w, h) for class_id=0 (person).
    """
    boxes = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                if class_id != 0:
                    continue

                cx, cy, w, h = map(float, parts[1:5])
                boxes.append((cx, cy, w, h))
    except FileNotFoundError:
        pass

    return boxes


def encode_targets(boxes: List[Tuple[float, float, float, float]], grid_size: int = 26) -> torch.Tensor:
    """
    Convert boxes to grid target tensor [S, S, 5] where last dim is [tx, ty, tw, th, obj].
    tx, ty are cell-relative offsets. Keeps first box if multiple map to same cell.
    """
    S = grid_size
    target = torch.zeros(S, S, 5, dtype=torch.float32)
    occupied = torch.zeros(S, S, dtype=torch.bool)

    for (cx, cy, w, h) in boxes:
        i = int(cx * S)
        j = int(cy * S)

        i = min(i, S - 1)
        j = min(j, S - 1)

        if occupied[j, i]:
            continue

        t_x = cx * S - i
        t_y = cy * S - j

        target[j, i, :] = torch.tensor([t_x, t_y, w, h, 1.0], dtype=torch.float32)
        occupied[j, i] = True

    return target
