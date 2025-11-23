#!/usr/bin/env python
"""
Training script for MLHD YOLO-like detector.
Usage: python scripts/train.py
"""
import sys
import os
from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.dataloader_augmented import CCTVDetectionDataset
from models.yolo_like import Model
from models.backbone import (
    BACKBONE_REGISTRY,
    backbone_output_grid,
    backbone_input_resolution,
)
from train.trainer import fit
from utils.device import get_best_device, describe_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLHD detector with selectable backbones.")
    parser.add_argument(
        "--backbone",
        default="13x13",
        choices=sorted(BACKBONE_REGISTRY.keys()),
        help="Backbone architecture to use (controls feature map size).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Hyperparameters (OPTIMIZED - Nov 2025)
    GRID_SIZE = backbone_output_grid(args.backbone)
    IMG_SIZE = backbone_input_resolution(args.backbone)
    BATCH_SIZE = 8  # Increased from 4 for more stable gradients
    EPOCHS = 50  # Increased from 100 for better convergence
    LR = 5e-4  # Increased from 5e-5 for faster convergence
    LAMBDA_COORD = 7.5 # Increased from 5.0 for better box localization
    LAMBDA_NOOBJ = 0.7  # Increased from 0.5 to reduce false positives
    EARLY_STOPPING_PATIENCE = 7  # Increased from 10 for more patience
    NUM_WORKERS = 0  # Use 0 for MPS on macOS (single-threaded data loading)
    MOMENTUM = 0.9
    # Paths
    TRAIN_IMAGES = ROOT / 'data/processed/training_5/images/train'
    TRAIN_LABELS = ROOT / 'data/processed/training_5/labels/train'
    VAL_IMAGES = ROOT / 'data/processed/training_5/images/val'
    VAL_LABELS = ROOT / 'data/processed/training_5/labels/val'
    CHECKPOINT_DIR = ROOT / 'checkpoints'
    DECAY = 5e-4
    # Device
    device = get_best_device()
    print(f"Using device: {describe_device(device)}")
    print(f"Selected backbone: {args.backbone} -> grid {GRID_SIZE}x{GRID_SIZE}, img {IMG_SIZE}x{IMG_SIZE}")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = CCTVDetectionDataset(
        image_dir=str(TRAIN_IMAGES),
        label_dir=str(TRAIN_LABELS),
        grid_size=GRID_SIZE,
        img_size=IMG_SIZE,
        augment=True  # Enable augmentation for training
    )

    val_dataset = CCTVDetectionDataset(
        image_dir=str(VAL_IMAGES),
        label_dir=str(VAL_LABELS),
        grid_size=GRID_SIZE,
        img_size=IMG_SIZE,
        augment=False  # No augmentation for validation
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\nInitializing model...")
    model = Model(S=GRID_SIZE, backbone_name=args.backbone)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, 
                                 weight_decay=DECAY)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training config
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LR}")
    print(f"Lambda coord: {LAMBDA_COORD}")
    print(f"Lambda noobj: {LAMBDA_NOOBJ}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print("="*60 + "\n")

    # Train
    print("Starting training...\n")
    results = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        lambda_coord=LAMBDA_COORD,
        lambda_noobj=LAMBDA_NOOBJ,
        obj_from_logits=False,
        scheduler=scheduler,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        ckpt_dir=str(CHECKPOINT_DIR),
        ckpt_name='best.pt',
        print_fn=print
    )

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation mIoU: {results['best_miou']:.4f}")
    print(f"Best validation loss at that point: {results['best_val']:.4f}")
    print(f"Checkpoint saved to: {CHECKPOINT_DIR / 'best.pt'}")
    print("="*60)


if __name__ == '__main__':
    main()
