# MLHD

From-scratch CNN for CCTV person detection and action recognition.

## Device Support

Training and evaluation scripts automatically select the best available accelerator. CUDA is preferred when present, followed by Apple Silicon's Metal Performance Shaders (MPS), then CPU as a fallback. No manual flag changes are required—`scripts/train.py` relies on `utils.device.get_best_device` to choose and report the active device.
