# MLHD Model Architecture Breakdown

## Overview
Your YOLO-like model is a lightweight CNN for single-class person detection in CCTV footage. It consists of:
1. **Backbone** (feature extraction)
2. **Detection Head** (prediction generation)
3. **Loss Function** (training objective)

---

## 1. Model Architecture

### Input → Output Flow
```
Input Image (3, 416, 416)
    ↓
Backbone (CNN)
    ↓
Feature Map (512, 26, 26)
    ↓
Detection Head (1×1 Conv + Resize)
    ↓
Predictions (26, 26, 5)
```

### Backbone Architecture
```python
# From models/backbone.py
Layer  | Input Channels | Output Channels | Stride | Output Size
-------|----------------|-----------------|--------|-------------
Conv1  | 3              | 16              | 2      | 208×208
Conv2  | 16             | 32              | 1      | 208×208
Conv3  | 32             | 64              | 2      | 104×104
Conv4  | 64             | 64              | 1      | 104×104
Conv5  | 64             | 128             | 2      | 52×52
Conv6  | 128            | 128             | 1      | 52×52
Conv7  | 128            | 256             | 2      | 26×26
Conv8  | 256            | 256             | 1      | 26×26
Conv9  | 256            | 512             | 1      | 26×26
```

**Key Features:**
- 9 convolutional layers
- 4 stride-2 downsampling layers (16× downsampling: 416→26)
- BatchNorm + LeakyReLU(0.1) after each conv
- No bias in conv layers (handled by BatchNorm)
- Total parameters: ~6.8M

### Detection Head
```python
# From models/head.py
1×1 Conv: 512 → 5 channels
Sigmoid activation
Bilinear interpolation to (26, 26)
Output: [tx, ty, tw, th, objectness]
```

---

## 2. Grid Cell System

### How Grid Cells Work
- Image divided into **S×S = 26×26 = 676 grid cells**
- Each cell size: 416/26 = **16×16 pixels**
- Each cell predicts **5 values**:
  - `tx, ty`: Offset within cell (0-1)
  - `tw, th`: Box width/height (normalized by image size)
  - `objectness`: Probability that cell contains object center

### Prediction Decoding
```python
# For cell (i, j):
cx = (i + tx) / S           # Absolute x-center (normalized)
cy = (j + ty) / S           # Absolute y-center (normalized)
w = tw                      # Width (normalized)
h = th                      # Height (normalized)

# Convert to pixels:
cx_pix = cx * 416
cy_pix = cy * 416
w_pix = w * 416
h_pix = h * 416

# Convert to corner format:
x1 = cx_pix - w_pix / 2
y1 = cy_pix - h_pix / 2
x2 = cx_pix + w_pix / 2
y2 = cy_pix + h_pix / 2
```

---

## 3. Loss Function

### Components
```python
Total Loss = λ_coord * (L_xy + L_wh) + L_obj

Where:
- L_xy = MSE on (tx, ty) offsets
- L_wh = MSE on squared sizes (tw², th²)
- L_obj = BCE on objectness
```

### Loss Weights (Current)
- **λ_coord = 5.0**: Coordinate loss weight
- **λ_noobj = 0.5**: Negative objectness weight

### Loss Calculation Details
```python
# Coordinate Loss (only for cells with objects)
L_xy = Σ [(px - tx)² + (py - ty)²]  for cells where obj=1
L_wh = Σ [(pw² - tw²)² + (ph² - th²)²]  for cells where obj=1

# Objectness Loss (all cells)
L_obj_pos = Σ BCE(pobj, 1)  for cells where obj=1
L_obj_neg = Σ BCE(pobj, 0) * λ_noobj  for cells where obj=0

L_obj = L_obj_pos + L_obj_neg
```

---

## 4. Current Hyperparameters

### Training Configuration (from scripts/train.py)
```python
IMG_SIZE = 416                    # Input image size
GRID_SIZE = 26                    # Grid cells per dimension
BATCH_SIZE = 4                    # Small batch for regularization
EPOCHS = 100                      # Training epochs
LR = 5e-5                         # Learning rate
LAMBDA_COORD = 5.0                # Coordinate loss weight
LAMBDA_NOOBJ = 0.5                # Negative objectness weight
EARLY_STOPPING_PATIENCE = 10      # Early stopping
```

### Optimizer & Scheduler
```python
Optimizer: Adam(lr=5e-5)
Scheduler: ReduceLROnPlateau(factor=0.5, patience=5)
```

---

## 5. Hyperparameter Tuning Guide

### Priority 1: Grid Size (S)

**Current: S=26**

**Impact:**
- Smaller S (7, 13) → Fewer cells, faster inference, worse localization
- Larger S (32, 52) → Better localization, slower inference, more memory

**Recommendations:**
```python
# For small objects (people far from camera):
S = 32 or S = 52  # Better spatial resolution

# For large objects (people close to camera):
S = 13  # Sufficient, faster

# Current S=26 is a balanced middle ground
```

**How to test:**
```python
# In train.py, change:
GRID_SIZE = 32  # or 13, 52

# And in Model:
model = Model(S=32)
```

---

### Priority 2: Loss Weights

**Current: λ_coord=5.0, λ_noobj=0.5**

**λ_coord (Coordinate Loss Weight)**
```python
# Problem: Boxes slightly off-center
λ_coord = 10.0  # Increase (more penalty for box errors)

# Problem: Model focuses too much on exact positions, poor detection
λ_coord = 2.0   # Decrease

# Recommended range: 2.0 - 10.0
```

**λ_noobj (Negative Objectness Weight)**
```python
# Problem: Too many false positives
λ_noobj = 0.8   # Increase (penalize false detections more)

# Problem: Missing objects (low recall)
λ_noobj = 0.3   # Decrease (be more lenient with background)

# Recommended range: 0.3 - 1.0
```

**How to test:**
```python
# In train.py:
LAMBDA_COORD = 10.0
LAMBDA_NOOBJ = 0.8

# Train and compare metrics
```

---

### Priority 3: Learning Rate & Batch Size

**Learning Rate**
```python
# Current: LR = 5e-5 (very conservative)

# For faster convergence:
LR = 1e-4  # 2x current rate

# If training unstable:
LR = 1e-5  # More stable

# Recommended: Use learning rate warmup
# Start low (1e-6), gradually increase to 1e-4 over first 5 epochs
```

**Batch Size**
```python
# Current: BATCH_SIZE = 4 (good regularization)

# For faster training (if you have GPU memory):
BATCH_SIZE = 8 or 16

# Note: Larger batch → adjust LR proportionally
# If batch doubles, consider doubling LR
```

---

### Priority 4: Backbone Depth

**Current Architecture:**
- 9 conv layers
- 512 output channels
- ~6.8M parameters

**Lightweight Option (Faster, Less Accurate):**
```python
# In models/backbone.py, reduce final channels:
conv_block(256, 256, k=3, s=1, p=1),  # Keep at 256 instead of 512

# And in head.py:
self.conv1x1 = nn.Conv2d(256, 5, ...)  # Change from 512
```

**Deeper Option (More Accurate, Slower):**
```python
# Add more layers after current backbone:
conv_block(512, 512, k=3, s=1, p=1),  # Additional layer
conv_block(512, 1024, k=3, s=1, p=1), # Go to 1024 channels

# Update head:
self.conv1x1 = nn.Conv2d(1024, 5, ...)
```

---

### Priority 5: Data Augmentation

**Current: None** ⚠️ **HIGH IMPACT**

**Recommended augmentations for CCTV:**
```python
import albumentations as A

augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.CLAHE(p=0.2),  # Histogram equalization
], bbox_params=A.BboxParams(format='yolo'))
```

**Expected impact: +5-15% mAP improvement**

---

## 6. Performance Bottlenecks

### Model Analysis
```
Current Performance:
- Validation Loss: 4.85
- Grid Size: 26×26 = 676 predictions per image
- Inference Speed: ~50ms per image (CPU)
- Model Size: ~26MB
```

### Bottlenecks:
1. **No data augmentation** → Overfitting risk
2. **Small batch size (4)** → Slow training
3. **Conservative LR (5e-5)** → Slow convergence
4. **Fixed grid size** → May miss small/large objects

---

## 7. Suggested Improvement Pipeline

### Phase 1: Quick Wins (No Retraining)
1. **Adjust confidence threshold** (try 0.3, 0.4, 0.6, 0.7)
2. **Adjust NMS IoU threshold** (try 0.3, 0.4, 0.6)
3. **Test different grid sizes** with current checkpoint

### Phase 2: Retrain with Better Hyperparameters
```python
# Recommended settings:
BATCH_SIZE = 8           # Increase if memory allows
LR = 1e-4               # Faster convergence
LAMBDA_COORD = 7.0      # Slightly more emphasis on localization
LAMBDA_NOOBJ = 0.7      # Reduce false positives
GRID_SIZE = 32          # Better spatial resolution
```

### Phase 3: Add Data Augmentation
- Implement Albumentations (see Priority 5 above)
- Expected: +10% mAP improvement

### Phase 4: Architecture Improvements
- Add skip connections (ResNet-style)
- Try deeper backbone (10-12 layers)
- Multi-scale predictions (predict at multiple resolutions)

---

## 8. Ablation Study Recommendations

Test these variations systematically:

### Grid Size Ablation
```python
for S in [13, 26, 32, 52]:
    train_model(grid_size=S)
    evaluate()
```

### Loss Weight Ablation
```python
coord_values = [2.0, 5.0, 7.0, 10.0]
noobj_values = [0.3, 0.5, 0.7, 1.0]

for λ_c in coord_values:
    for λ_n in noobj_values:
        train_model(lambda_coord=λ_c, lambda_noobj=λ_n)
        evaluate()
```

### Learning Rate Ablation
```python
lr_values = [1e-5, 5e-5, 1e-4, 5e-4]
for lr in lr_values:
    train_model(learning_rate=lr)
    evaluate()
```

---

## 9. Expected Performance Improvements

| Change | Expected mAP Gain | Training Time Impact |
|--------|-------------------|---------------------|
| Add data augmentation | +5-15% | +20% (more epochs needed) |
| Increase grid size (26→32) | +2-5% | +15% (more parameters) |
| Optimize loss weights | +2-8% | None |
| Increase batch size (4→8) | +1-3% | -30% (faster) |
| Increase LR (5e-5→1e-4) | +2-5% | -20% (faster convergence) |
| Deeper backbone (+2 layers) | +3-7% | +25% (slower training) |

**Combined (augmentation + tuning):** +15-25% mAP improvement

---

## 10. Quick Reference: Where to Change What

```python
# Grid size:
scripts/train.py: GRID_SIZE = 32
models/yolo_like.py: Model(S=32)

# Loss weights:
scripts/train.py: LAMBDA_COORD = 7.0, LAMBDA_NOOBJ = 0.7

# Learning rate:
scripts/train.py: LR = 1e-4

# Batch size:
scripts/train.py: BATCH_SIZE = 8

# Backbone depth:
models/backbone.py: Add/remove conv_block layers

# Data augmentation:
datasets/dataloader.py: Add augmentation pipeline
```

---

## Summary

Your model is a **well-structured, lightweight YOLO-like detector** with:
- ✅ Clean architecture (Backbone + Head)
- ✅ Appropriate loss function for single-class detection
- ✅ MPS/CUDA support for Apple Silicon
- ⚠️ No data augmentation (biggest opportunity)
- ⚠️ Conservative training hyperparameters

**Top 3 Recommendations:**
1. **Add data augmentation** (Albumentations) → +10-15% mAP
2. **Increase learning rate to 1e-4** → Faster convergence
3. **Test grid size 32** → Better localization for small objects
