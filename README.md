# MLHD

From-scratch YOLO-like detector for CCTV person detection and action recognition.

## Contents you actually have
- Model code: `models/backbone.py`, `models/head.py`, `models/yolo_like.py`
- Training loop: `scripts/train.py` (uses `datasets/dataloader_augmented.py` and `train/trainer.py`)
- Inference tools: `scripts/infer.py`, `scripts/infer_video.py`
- Utilities: `utils/make_mini_dataset.py`, `utils/batch_compare_predictions.py`, `eval/latency_benchmark.py`, `eval/label_counts.py`
- Expected weights in repo root: `checkpoints/best.pt` (our model), `yolov5s.pt` (reference)

## Setup
```
pip install -r requirements.txt
```
Place `checkpoints/best.pt` and `yolov5s.pt` in the repo root (paths are hardcoded in several scripts).

## Dataset assumptions (you only have a mini set)
- The codebase assumes a dataset rooted at `data/processed/training_5`.
- `scripts/train.py` hardcodes these paths:
  - `TRAIN_IMAGES = data/processed/training_5/images/train`
  - `TRAIN_LABELS = data/processed/training_5/labels/train`
  - `VAL_IMAGES   = data/processed/training_5/images/val`
  - `VAL_LABELS   = data/processed/training_5/labels/val`
- If you only have a mini dataset, either:
  1) Put your mini set at `data/processed/training_5` (rename/copy/symlink), keeping the layout:
     ```
     data/processed/training_5/
       images/train/*.jpg
       images/val/*.jpg
       labels/train/*.txt  # YOLO: class cx cy w h
       labels/val/*.txt
     ```
  2) Or edit `scripts/train.py` to point `TRAIN_*` and `VAL_*` to your mini root (e.g., `data/processed/training_5_mini`). There are no CLI flags for dataset paths in `train.py`.

## Making a mini dataset from your own data
Use the helper to carve out a subset with the required layout:
```
python -m utils.make_mini_dataset \
  --source <your_full_dataset_root> \
  --dest data/processed/training_5 \
  --train-count 100 \
  --val-count 40
```
This copies images/labels into the destination, skipping images without labels and warning if fewer than requested are found. If you use a different `--dest`, update the hardcoded paths in scripts accordingly.

## Training (only CLI flag is --backbone)
Edit `scripts/train.py` if your data is not at `data/processed/training_5`. After paths are correct:
```
python scripts/train.py --backbone 13x13
```
Backbone options: `13x13` or `8x8` (sets grid/input size). Device is auto-selected (CUDA > MPS > CPU) via `utils.device.get_best_device`.

## Inference
- Single image:
  ```
  python scripts/infer.py \
    --checkpoint checkpoints/best.pt \
    --image data/processed/training_5/images/val/example.jpg \
    --grid-size 13 \
    --img-size 416
  ```
  Point `--image` to your file if your data lives elsewhere.
- Video/webcam:
  ```
  python scripts/infer_video.py \
    --checkpoint checkpoints/best.pt \
    --source path/to/video.mp4 \
    --backbone 13x13
  ```

## Visual comparison vs YOLOv5s
```
python -m utils.batch_compare_predictions \
  --images-dir data/processed/training_5/images/val \
  --mlhd-checkpoint checkpoints/best.pt \
  --yolo-weights yolov5s.pt \
  --output-dir outputs/predictions
```
Change `--images-dir` if your validation images live elsewhere.

## Latency benchmark
`eval/latency_benchmark.py` uses constants at the top:
- `IMAGES_DIR` (defaults to `data/processed/training_5/images/val`)
- `MLHD_CKPT` (defaults to `checkpoints/best.pt`)
- `YOLO_WEIGHTS` (defaults to `yolov5s.pt`)
Edit these to match your setup, then run:
```
python -m eval.latency_benchmark --warmup 2 --max-images 200
```
Outputs: `outputs/latency_benchmark/{latency_metrics.json, latency_boxplot.png, latency_histogram.png}` and printed stats.

## Label counting
`eval/label_counts.py` counts files/boxes under `data/processed/training_5/labels` by default. Change `LABEL_ROOT` inside if your dataset root differs, then:
```
python -m eval.label_counts
```

## Using your own dataset (summary)
1) Arrange data as YOLO format:
   ```
   <root>/images/{train,val}/*.jpg
   <root>/labels/{train,val}/*.txt   # class cx cy w h
   ```
2) Either place it at `data/processed/training_5` or edit hardcoded paths in:
   - `scripts/train.py` (`TRAIN_*`, `VAL_*`, `CHECKPOINT_DIR`)
   - `eval/latency_benchmark.py` (`IMAGES_DIR`, `MLHD_CKPT`, `YOLO_WEIGHTS`)
   - `eval/label_counts.py` (`LABEL_ROOT`)
3) For scripts that accept CLI paths (`infer.py`, `infer_video.py`, `utils.batch_compare_predictions.py`), pass your files/dirs via flags.

## Device support
Training and evaluation scripts automatically select the best available accelerator. Preference order: CUDA, then Apple MPS, then CPU. Selection is handled by `utils.device.get_best_device`.
