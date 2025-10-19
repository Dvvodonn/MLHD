import torch
from datasets.target_encoding import load_yolo_labels, encode_targets
from datasets.transforms import transform_boxes

print("=" * 50)
print("Testing Data Pipeline Components")
print("=" * 50)

label_path = "data/raw/training_3/labels/train/000303.txt"
print(f"\n1. Testing load_yolo_labels with {label_path}")
boxes = load_yolo_labels(label_path)
print(f"   Loaded {len(boxes)} boxes:")
for i, (cx, cy, w, h) in enumerate(boxes[:3]):
    print(f"   Box {i}: cx={cx:.4f}, cy={cy:.4f}, w={w:.4f}, h={h:.4f}")

print(f"\n2. Testing encode_targets with grid_size=13")
target = encode_targets(boxes, grid_size=13)
print(f"   Target shape: {target.shape}")
print(f"   Cells with objects: {(target[:, :, 0] > 0).sum().item()}")

print(f"\n3. Inspecting target tensor for first box")
box_0 = boxes[0]
i = int(box_0[0] * 13)
j = int(box_0[1] * 13)
print(f"   First box (cx={box_0[0]:.4f}, cy={box_0[1]:.4f}) -> cell [{j}, {i}]")
print(f"   Target[{j}, {i}] = {target[j, i].tolist()}")

print(f"\n4. Testing letterbox coordinate transformation (mock params)")
mock_params = {
    'scale': 0.5,
    'pad_w': 50,
    'pad_h': 50,
    'orig_wh': (800, 600),
    'target_wh': (416, 416)
}
transformed = transform_boxes(boxes[:2], mock_params)
print(f"   Original box 0: {boxes[0]}")
print(f"   Transformed:    {transformed[0]}")

print(f"\n5. Testing full Dataset (will fail if images are broken symlinks)")
try:
    from datasets.dataloader import CCTVDetectionDataset
    dataset = CCTVDetectionDataset(
        image_dir="data/raw/training_3/images/train",
        label_dir="data/raw/training_3/labels/train",
        grid_size=13,
        img_size=416
    )
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Attempting to load first sample...")
    image, target = dataset[0]
    print(f"   SUCCESS! Image shape: {image.shape}, Target shape: {target.shape}")
except Exception as e:
    print(f"   EXPECTED ERROR (images are symlinks): {e}")

print("\n" + "=" * 50)
print("Pipeline test complete!")
print("=" * 50)
