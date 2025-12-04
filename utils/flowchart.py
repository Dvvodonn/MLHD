import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# Create figure
fig, ax = plt.subplots(figsize=(14, 3))
ax.set_xlim(0, 14)
ax.set_ylim(0, 3)
ax.axis('off')

def draw_box(x, text):
    rect = Rectangle((x, 1), 2.5, 1, edgecolor='black', facecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + 1.25, 1.5, text, ha='center', va='center', fontsize=10, wrap=True)

def draw_arrow(x1, x2):
    arrow = FancyArrowPatch((x1, 1.5), (x2, 1.5), arrowstyle='->', mutation_scale=15, linewidth=2)
    ax.add_patch(arrow)

# Draw boxes
texts = [
    "Surveillance Platform Videos",
    "Frame Extraction Scripts",
    "Pre-existing Detectors\nGenerate Labels",
    "LabelMe Verification & Cleanup",
    "Train/Test Split\n86% / 14%"
]

positions = [0.5, 3.5, 6.5, 9.5, 12.0]

for pos, text in zip(positions, texts):
    draw_box(pos, text)

# Arrows
for i in range(len(positions) - 1):
    draw_arrow(positions[i] + 2.5, positions[i + 1])

# Save output
plt.savefig("mlhd_data_flow.png", dpi=300, bbox_inches='tight')