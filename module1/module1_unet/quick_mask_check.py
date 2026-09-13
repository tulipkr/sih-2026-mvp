import rasterio
import numpy as np
from pathlib import Path

mask_dir = Path(r"C:\Users\tulip\OneDrive\Desktop\SIH\datasets\Mask_oil")

masks = sorted(mask_dir.glob("*.tif"))

print("Total masks:", len(masks))

sample_indices = [
    0, 99, 199, 299, 399, 499,
    599, 699, 799, 899, 999, 1099, 1199
]

scenes_with_oil = 0
total_oil_pixels = 0
total_pixels = 0

print("\nChecking 13 sample scenes...\n")

for i in sample_indices:
    with rasterio.open(masks[i]) as src:
        mask = src.read(1)

    oil_pixels = np.sum(mask == 1)

    if oil_pixels > 0:
        scenes_with_oil += 1

    total_oil_pixels += oil_pixels
    total_pixels += mask.size

    print(
        f"{masks[i].name}: "
        f"oil pixels = {oil_pixels:,}, "
        f"oil fraction = {oil_pixels / mask.size:.6f}"
    )

print("\n--- SUMMARY ---")
print(f"Sample scenes containing oil: {scenes_with_oil} / {len(sample_indices)}")
print(f"Sample oil-pixel fraction: {total_oil_pixels / total_pixels:.6f}")