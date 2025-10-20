# MLHD Project Roadmap

## Overall Progress: 85% Complete

**Last Updated:** 2025-10-20
**Status:** ✅ Model Trained & Inference Working! Can now visualize predictions.

---

## Phase 1: Data Pipeline (COMPLETE ✅)

### 1.1 Data Loading & Parsing
- [x] Load YOLO format labels (`datasets/target_encoding.py::load_yolo_labels()`)
- [x] Load images from disk (`frame_resize/image_to_numpy.py`)
- [x] Letterbox resize with aspect ratio (`frame_resize/resize.py`)

### 1.2 Data Transforms
- [x] Image letterbox wrapper (`datasets/transforms.py::letterbox_image()`)
- [x] Box coordinate adjustment for padding (`datasets/transforms.py::transform_boxes()`)
- [x] Image normalization to [0,1]

### 1.3 Target Encoding
- [x] Grid cell assignment (S×S grid)
- [x] Cell-relative offset computation (tx, ty)
- [x] Target tensor creation [S, S, 5] format: [obj, tx, ty, tw, th]
- [x] Collision handling (keep first box)

### 1.4 Dataset & DataLoader
- [x] PyTorch Dataset class (`datasets/dataloader.py::CCTVDetectionDataset`)
- [x] DataLoader integration (batching, shuffling)
- [x] Output format: images [B,3,416,416], targets [B,13,13,5]

**Status:** READY FOR CNN TRAINING ✅

---

## Phase 2: Model Architecture (COMPLETE ✅)

### 2.1 CNN Backbone
- [x] Convolutional layers: 16→32→64→128→256→512 channels (31 lines)
- [x] Stride-2 downsampling (4 stride-2 layers)
- [x] LeakyReLU activation (0.1 negative slope)
- [x] Batch normalization on all layers
- [x] Output: [B, 512, 26, 26] feature map
- **File:** `models/backbone.py` ✅

**Architecture:**
```
3→16 (s=2) → 16→32 → 32→64 (s=2) → 64→64 → 64→128 (s=2) →
128→128 → 128→256 (s=2) → 256→256 → 256→512
```

### 2.2 Detection Head
- [x] 1×1 convolution: 512 → 5 channels (22 lines)
- [x] Sigmoid activation for all outputs
- [x] Bilinear interpolation to resize to [S, S]
- [x] Output: [B, S, S, 5] permuted format
- **File:** `models/head.py` ✅

### 2.3 Full YOLO-like Model
- [x] Combine backbone + head (14 lines)
- [x] Forward pass implementation
- [x] Configurable grid size S (default: 26)
- **File:** `models/yolo_like.py` ✅

---

## Phase 3: Loss Function (COMPLETE ✅)

### 3.1 Localization Loss
- [x] MSE on (tx, ty) offsets (101 lines total)
- [x] MSE on (tw², th²) squared sizes
- [x] Apply only to cells with objects (obj=1)
- [x] Weight with λ_coord hyperparameter (default: 5.0)

### 3.2 Objectness Loss
- [x] Binary cross-entropy for objectness
- [x] Positive cells: BCE(pred_obj, 1)
- [x] Negative cells: BCE(pred_obj, 0) weighted by λ_noobj (default: 0.5)
- [x] Sum over all S×S cells
- [x] Support for both sigmoid probabilities and logits

### 3.3 Combined Loss
- [x] L = λ_coord * (L_xy + L_wh) + L_obj
- [x] Normalized by batch size
- [x] Implemented in `losses/detection_loss.py` ✅

**Key Features:**
- Coordinate loss: `λ_coord * [(px-tx)² + (py-ty)² + (pw²-tw²)² + (ph²-th²)²]`
- Objectness loss: `BCE(pos) + λ_noobj * BCE(neg)`
- Flexible obj_from_logits flag for training stability

---

## Phase 4: Training (COMPLETE ✅)

### 4.1 Trainer Class
- [x] Training loop (epochs, batches) (210 lines)
- [x] Forward pass
- [x] Loss computation
- [x] Backward pass + optimization
- [x] Validation loop
- [x] Multi-epoch fit() function with checkpointing
- **File:** `train/trainer.py` ✅

### 4.2 Optimization
- [x] Adam optimizer with learning rate 1e-4
- [x] Learning rate scheduler (ReduceLROnPlateau)
- [x] Configurable hyperparameters (λ_coord=5.0, λ_noobj=0.5)

### 4.3 Training Script
- [x] Main training script (139 lines) ✅
- [x] Dataset setup (CCTVDetectionDataset)
- [x] Model initialization (S=26 grid)
- [x] MPS device support (Apple Silicon GPU)
- [x] Data conversion (LabelMe → YOLO format)
- **File:** `scripts/train.py` ✅
- **File:** `scripts/labelme_to_yolo.py` ✅

### 4.4 Training Results
- [x] **Model trained successfully!**
- [x] 50 epochs completed
- [x] Best validation loss: **4.8490**
- [x] Checkpoint saved: `checkpoints/best.pt`
- [x] Train/val split: 1958/218 images

### 4.5 Metrics & Logging
- [x] Training loss tracking (per epoch)
- [x] Validation loss tracking (per epoch)
- [x] Best checkpoint saving
- [ ] IoU calculation for predicted boxes ❌ **NEXT PRIORITY**
- [ ] Tensorboard logging ❌

---

## Phase 5: Inference & Visualization (COMPLETE ✅)

**Status:** Inference script implemented and working

### 5.1 Inference Pipeline ✅ **COMPLETE**
- [x] Load trained model checkpoint (`checkpoints/best.pt`)
- [x] Process single image
- [x] Decode predictions from grid [S,S,5] → bounding boxes
- [x] Non-maximum suppression (NMS)
- [x] Confidence thresholding
- **File:** `scripts/infer.py` (304 lines)

**Usage:**
```bash
python scripts/infer.py --image path/to/image.jpg --conf 0.6 --output result.jpg
```

**Note:** Model produces many false positives at image edges. Recommend tuning confidence threshold or additional training.

### 5.2 Basic Visualization ✅ **COMPLETE**
- [x] Draw predicted bounding boxes on images
- [x] Save output images with predictions
- [ ] Compare ground truth vs predictions (optional)
- [ ] Grid cell activation heatmaps (optional)
- **File:** `scripts/infer.py` (includes visualization)

### 5.3 Utilities ✅ **COMPLETE**
- [x] `decode_predictions()` - Convert grid [S,S,5] → boxes [(x1,y1,x2,y2,conf), ...]
- [x] `nms()` - Remove duplicate detections
- [x] `draw_boxes()` - Visualize boxes on image
- [x] `transform_boxes_to_original()` - Map letterbox coords to original image

---

## Phase 6: Evaluation Metrics (PENDING ❌)

### 6.1 Metrics
- [ ] IoU calculation on validation set
- [ ] Precision/Recall curves
- [ ] mAP (mean Average Precision)
- **File:** `eval/evaluator.py`

### 6.2 Analysis
- [ ] Training/validation loss curves plot
- [ ] Per-class performance breakdown
- [ ] Failure case analysis

---

## Phase 7: Experiments & Comparison (PENDING ❌)

### 7.1 Baseline
- [ ] Train baseline model
- [ ] Record IoU on validation set

### 7.2 YOLOv4 Comparison
- [ ] Run YOLOv4 on same dataset
- [ ] Compare IoU scores
- [ ] Document domain shift performance

### 7.3 Ablation Studies
- [ ] Grid size experiments (S=7, 13, 26)
- [ ] Loss weight tuning (λ_coord, λ_noobj)
- [ ] Backbone depth experiments

---

## Utilities (Supporting)

- [ ] `utils/seeding.py` - Reproducibility
- [ ] `utils/logging.py` - Training logs
- [ ] `utils/io.py` - Checkpoint save/load
- [ ] `utils/timing.py` - Performance profiling

---

## Timeline (From Proposal)

- **Weeks 1-3:** Data prep, grid encoding, baseline model ✅ **COMPLETE**
- **Weeks 4-6:** Training and hyperparameter search ✅ **COMPLETE**
  - Model trained with val loss 4.85
- **Weeks 7-8:** Evaluation, visualization, final report ⚠️ **IN PROGRESS**

---

## Current Status Summary

| Component | Status | Files | Lines |
|-----------|--------|-------|-------|
| Data Pipeline | ✅ DONE | `datasets/*.py`, `frame_resize/*.py` | ~174 |
| Model Architecture | ✅ DONE | `models/*.py` (backbone, head, yolo_like) | 67 |
| Loss Function | ✅ DONE (MPS-compatible) | `losses/detection_loss.py` | 101 |
| Training Loop | ✅ DONE | `train/trainer.py` | 210 |
| Training Script | ✅ DONE | `scripts/train.py` | 139 |
| Data Conversion | ✅ DONE | `scripts/labelme_to_yolo.py` | 246 |
| **Model Trained** | ✅ DONE | `checkpoints/best.pt` | - |
| **Inference Script** | ✅ DONE | `scripts/infer.py` | 304 |
| **Basic Visualization** | ✅ DONE | `scripts/infer.py` (included) | - |
| Evaluation Metrics | ❌ TODO | `eval/evaluator.py` (empty) | 0 |

**Total Code Written:** ~1,241 lines

**Next Priority:** Evaluation metrics (IoU, mAP) for quantitative assessment

---

## Key Implementation Decisions Made

1. **Grid size:** S=26 (16×16 pixels per cell for 416×416 images) - changed from initial S=13
2. **Target format:** [obj, tx, ty, tw, th] (objectness first)
3. **Collision handling:** Keep first box if multiple map to same cell
4. **Image transform:** Letterbox resize (preserves aspect ratio)
5. **Coordinate adjustment:** Transform boxes for letterbox padding
6. **Framework:** PyTorch with standard nn.Module (backbone uses BatchNorm + LeakyReLU)
7. **Loss weights:** λ_coord=5.0, λ_noobj=0.5 (defaults)
8. **Activation:** Sigmoid on all detection head outputs

---

## Notes

- Images in `data/raw/training_3/` are broken symlinks - need actual image files
- 1958 training samples, 218 validation samples
- Single class detection (person, class_id=0)
- PyTorch DataLoader tested and working
