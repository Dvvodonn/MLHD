import torch
from models.yolo_like import Model
from losses.detection_loss import detection_loss

print("=" * 60)
print("Testing MPS Compatibility")
print("=" * 60)

# Check MPS
print(f"\nMPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if not torch.backends.mps.is_available():
    print("MPS not available, exiting")
    exit(0)

device = torch.device('mps')
print(f"Using device: {device}")

# Test model
print("\n1. Testing model on MPS...")
try:
    model = Model(S=26).to(device)
    print("   ✓ Model moved to MPS")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 416, 416, device=device)
    output = model(dummy_input)
    print(f"   ✓ Forward pass: input {dummy_input.shape} → output {output.shape}")

except Exception as e:
    print(f"   ✗ Model failed: {e}")
    exit(1)

# Test loss
print("\n2. Testing loss on MPS...")
try:
    preds = torch.rand(2, 26, 26, 5, device=device)
    targets = torch.zeros(2, 26, 26, 5, device=device)
    targets[0, 5, 5, 0] = 1.0  # One object

    loss = detection_loss(preds, targets)
    print(f"   ✓ Loss computed: {loss.item():.4f}")

except Exception as e:
    print(f"   ✗ Loss failed: {e}")
    exit(1)

# Test backward with proper gradients
print("\n3. Testing backward pass on MPS...")
try:
    # Create a fresh forward pass with model
    dummy_input = torch.randn(2, 3, 416, 416, device=device)
    output = model(dummy_input)

    # Create target
    target = torch.zeros(2, 26, 26, 5, device=device)
    target[0, 5, 5, 0] = 1.0

    # Compute loss and backward
    loss = detection_loss(output, target)
    loss.backward()
    print("   ✓ Backward pass successful")
    print(f"   ✓ Loss value: {loss.item():.4f}")

except Exception as e:
    print(f"   ✗ Backward failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("MPS Compatibility: ALL TESTS PASSED ✓")
print("=" * 60)
