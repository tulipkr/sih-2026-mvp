from pathlib import Path
import rasterio

mask_dir = Path(r"C:\Users\tulip\OneDrive\Desktop\SIH\datasets\Mask_oil")

patch_size = 256
positive_tiles = 0
total_tiles = 0

for mask_path in sorted(mask_dir.glob("*.tif")):
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    height, width = mask.shape

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            tile = mask[y:y + patch_size, x:x + patch_size]

            if tile.shape != (patch_size, patch_size):
                continue

            total_tiles += 1

            if tile.any():
                positive_tiles += 1

print(f"Total tiles: {total_tiles}")
print(f"Positive tiles: {positive_tiles}")
print(f"Positive fraction: {positive_tiles / total_tiles:.3f}")