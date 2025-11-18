# Model Performance Diagnosis

## Current Results Summary

```
Overall Metrics (IoU 0.5):
- Precision: 53.62%  (393 TP / 733 predictions)
- Recall: 52.19%     (393 TP / 753 GT objects)
- F1 Score: 52.89%
- mAP@0.5: 30.25%

Overall Metrics (IoU 0.75):
- Precision: 11.46%  (84 TP / 733 predictions)
- Recall: 11.16%     (84 TP / 753 GT objects)
- F1 Score: 11.31%
- mAP@0.75: 2.40%

Dataset:
- 218 validation images
- 753 ground truth objects
- 733 predicted objects (close to GT count!)
```

---

## 🔴 Critical Issues Identified

### Issue #1: MASSIVE Drop from IoU@0.5 to IoU@0.75
**What this means:**
- At IoU 0.5: 53% precision/recall → Model detects objects but boxes are **imprecise**
- At IoU 0.75: 11% precision/recall → Only 84 out of 393 boxes are well-localized

**Root Cause:**
```
The model finds the right general area but struggles with exact bounding box coordinates.

IoU 0.5 requires only 50% overlap → Easy to pass
IoU 0.75 requires 75% overlap → Your boxes are too loose
```

**Visual Example:**
```
Ground Truth:  [████████]
Your Pred:     [██████████]    ← IoU = 0.6 (passes 0.5, fails 0.75)
Good Pred:     [████████]      ← IoU = 0.9 (passes both)
```

---

### Issue #2: Balanced FP/FN but Mediocre Precision
**Numbers:**
- False Positives: 340 (predicting objects that don't exist)
- False Negatives: 360 (missing real objects)
- Predictions: 733 ≈ Ground Truth: 753 ✓

**What this means:**
- Model predicts roughly the correct number of objects ✓
- BUT: 340 are wrong locations (FP) and 360 real objects are missed (FN)
- The model is **guessing** well but not **detecting** well

---

### Issue #3: Inconsistent Performance Across Images
**Good images:** 12 images with perfect F1=1.0
**Bad images:** Many with F1<0.5

**Pattern Analysis:**
- Best: Simple scenes with 1-7 people, clear views
- Worst: Complex scenes, occlusions, small people, edge cases

---

## 🎯 Root Causes (In Priority Order)

### 1. **Poor Localization (Box Regression)**
**Evidence:**
- mAP@0.5 = 30% → mAP@0.75 = 2.4% (12.5× drop!)
- Only 84/393 predictions have IoU ≥ 0.75

**Why:**
- Current λ_coord = 5.0 is too low
- Model optimizes more for objectness than precise localization
- No penalty for oversized/undersized boxes beyond MSE

**Fix:**
```python
# Increase coordinate loss weight
LAMBDA_COORD = 10.0  # or even 15.0

# This forces model to care more about exact box positions
```

---

### 2. **Training Loss Too High (4.85)**
**Evidence:**
- Validation loss: 4.85 (not converged)
- Model was trained with very conservative settings:
  - LR = 5e-5 (very slow)
  - Batch size = 4 (very small)
  - Early stopping at epoch ~50

**Why:**
- Model didn't train long enough
- Learning rate too low → slow convergence
- Small batch size → noisy gradients

**Fix:**
```python
# Retrain with:
LR = 1e-4              # 2x faster
BATCH_SIZE = 8         # More stable gradients
EPOCHS = 150           # Allow more training
EARLY_STOPPING_PATIENCE = 20  # More patience
```

---

### 3. **No Data Augmentation**
**Evidence:**
- CCTV has variable lighting, angles, scales
- Model trained on raw images only
- Likely overfitting to training set appearance

**Why:**
- Real CCTV has:
  - Different lighting conditions
  - Motion blur
  - Varying scales (near/far people)
  - Different camera angles
- Training data doesn't capture this diversity

**Fix:**
```python
# Add augmentation (see MODEL_ARCHITECTURE.md)
import albumentations as A

augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
], bbox_params=A.BboxParams(format='yolo'))
```

---

### 4. **Grid Size May Be Suboptimal**
**Evidence:**
- Current S = 26 (each cell = 16×16 pixels)
- For people at various scales, this might be:
  - Too coarse for small/distant people
  - Fine for medium/close people

**Why:**
- 16×16 pixel cells mean:
  - Small people (< 32 pixels) hard to detect
  - Large people (> 100 pixels) might span multiple cells

**Fix:**
```python
# Try finer grid for better small object detection
GRID_SIZE = 32  # 13×13 pixel cells

# Or even finer:
GRID_SIZE = 52  # 8×8 pixel cells (YOLO standard)
```

---

### 5. **Objectness vs Localization Imbalance**
**Evidence:**
- Model finds objects (similar pred/GT counts)
- But boxes are imprecise (huge IoU drop)

**Why:**
- Loss function: `Total = λ_coord * (L_xy + L_wh) + L_obj`
- Current: λ_coord=5.0, λ_noobj=0.5
- Objectness loss dominates early in training
- Model learns "where" before "exactly where"

**Fix:**
```python
# Rebalance losses
LAMBDA_COORD = 10.0   # 2x importance on precise boxes
LAMBDA_NOOBJ = 0.7    # Slightly higher to reduce FPs

# Or use curriculum learning:
# Start: λ_coord=5.0, then gradually increase to 15.0
```

---

## 📊 Detailed Breakdown by Numbers

### What Your Model Does Well:
✅ **Object Count Estimation**: 733 pred vs 753 GT (97% accurate count)
✅ **General Detection**: 53% recall means it finds half the objects
✅ **Some Perfect Images**: 12 images with F1=1.0 (easy cases)

### What Your Model Does Poorly:
❌ **Precise Localization**: mAP@0.75 only 2.4%
❌ **False Positives**: 340 wrong detections (46% of predictions)
❌ **Missed Objects**: 360 objects not detected (48% of GT)
❌ **Consistency**: High variance across images

---

## 🚨 Problem Severity Ranking

| Issue | Impact | Difficulty to Fix | Priority |
|-------|--------|-------------------|----------|
| Poor box localization | ⭐⭐⭐⭐⭐ | Easy (adjust λ_coord) | 🔥 HIGH |
| No data augmentation | ⭐⭐⭐⭐ | Medium (add albumentations) | 🔥 HIGH |
| Training not converged | ⭐⭐⭐⭐ | Easy (retrain longer) | 🔥 HIGH |
| Grid size suboptimal | ⭐⭐⭐ | Easy (change S) | MEDIUM |
| Loss imbalance | ⭐⭐⭐ | Easy (tune weights) | MEDIUM |

---

## 🎯 Action Plan (Prioritized)

### Quick Wins (No Retraining)
1. **Lower confidence threshold** → Test with 0.3, 0.4
   - Might improve recall at cost of precision
2. **Adjust NMS threshold** → Test with 0.3, 0.4, 0.6
   - Might reduce duplicate detections

### Short-Term Fixes (Retrain Required)
1. **Increase λ_coord to 10.0** → Better box localization
2. **Increase learning rate to 1e-4** → Faster convergence
3. **Train longer (150 epochs)** → Better convergence
4. **Increase batch size to 8** → More stable training

**Expected improvement: +15-25% mAP@0.5, +5-10% mAP@0.75**

### Medium-Term Improvements
1. **Add data augmentation** → Better generalization
2. **Increase grid size to 32** → Better small object detection
3. **Tune λ_noobj to 0.7** → Reduce false positives

**Expected improvement: +20-30% mAP@0.5, +10-15% mAP@0.75**

### Long-Term Improvements
1. **Deeper backbone** → More capacity
2. **Multi-scale predictions** → Handle various object sizes
3. **Focal loss** → Better handling of hard examples

**Expected improvement: +30-40% mAP@0.5, +15-20% mAP@0.75**

---

## 📈 Expected Performance After Fixes

### Current:
```
mAP@0.5: 30.25%
mAP@0.75: 2.40%
Precision: 53.62%
Recall: 52.19%
```

### After Quick Fixes (λ_coord=10, LR=1e-4):
```
mAP@0.5: 45-50%  (+15-20%)
mAP@0.75: 8-12%  (+6-10%)
Precision: 60-65%
Recall: 60-65%
```

### After Adding Augmentation:
```
mAP@0.5: 55-60%  (+25-30%)
mAP@0.75: 15-20% (+13-18%)
Precision: 65-70%
Recall: 65-70%
```

### After All Fixes (Including Architecture):
```
mAP@0.5: 65-75%  (+35-45%)
mAP@0.75: 25-35% (+23-33%)
Precision: 70-75%
Recall: 70-75%
```

---

## 🔍 Specific Examples to Investigate

### Best Performing Images (F1=1.0):
- 000282.jpg, 000454.jpg, 001126.jpg, etc.
- **Why they work:** Likely simple scenes, good lighting, clear views
- **Lesson:** Model works well in ideal conditions

### Worst Performing Images (Check outputs):
- Look for images with F1 < 0.3
- **Common patterns:**
  - Occlusions (people overlapping)
  - Small/distant people
  - Edge of frame detections
  - Poor lighting

---

## 💡 Key Insights

### The Good News:
1. Your model architecture is sound
2. It finds the right number of objects
3. Some images work perfectly
4. The issues are fixable with hyperparameter tuning

### The Bad News:
1. Boxes are too imprecise (huge IoU@0.75 drop)
2. Training didn't converge (loss 4.85 is high)
3. No augmentation = poor generalization
4. Inconsistent performance across scenes

### The Bottom Line:
**Your model is undertrained and under-tuned.**
It has the right capacity but needs:
- Better training (longer, faster LR, augmentation)
- Better loss weighting (higher λ_coord)
- Possibly better grid size (try 32 or 52)

---

## 🚀 Recommended Next Steps

### Step 1: Run Quick Diagnostic (5 minutes)
```bash
# Test different confidence thresholds
python scripts/evaluate.py --conf 0.3
python scripts/evaluate.py --conf 0.4
python scripts/evaluate.py --conf 0.6
```

### Step 2: Retrain with Better Settings (2-4 hours)
```python
# Edit scripts/train.py:
BATCH_SIZE = 8
LR = 1e-4
LAMBDA_COORD = 10.0
LAMBDA_NOOBJ = 0.7
EPOCHS = 150
EARLY_STOPPING_PATIENCE = 20

# Then run:
python scripts/train.py
```

### Step 3: Add Augmentation (1-2 hours coding + retrain)
- Implement albumentations in datasets/dataloader.py
- Retrain with augmentation
- Expected: +10-15% mAP improvement

### Step 4: Experiment with Grid Size
```python
# Try S=32
GRID_SIZE = 32
# Retrain and compare
```

---

## Summary

**Main Problem:** Your model finds objects but boxes are imprecise.

**Root Causes:**
1. λ_coord too low (5.0) → boxes not penalized enough
2. Training not converged (LR too low, stopped too early)
3. No augmentation → overfitting to training appearance
4. Possible grid size mismatch for object scales

**Quick Fix:** Retrain with λ_coord=10.0, LR=1e-4, longer training
**Expected Improvement:** +15-25% mAP@0.5, +5-10% mAP@0.75

**Your model has potential - it just needs better training!** 🚀
