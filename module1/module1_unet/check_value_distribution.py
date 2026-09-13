import rasterio
import numpy as np
from pathlib import Path

oil_dir = Path(r"C:\Users\tulip\OneDrive\Desktop\SIH\datasets\Oil")
files = sorted(oil_dir.glob("*.tif"))

# Check 50 evenly spaced images
indices = np.linspace(0, len(files) - 1, 50, dtype=int)

vv_values = []
vh_values = []

print(f"Total images: {len(files)}")
print("Checking 50 representative images...")

for i in indices:
    with rasterio.open(files[i]) as src:
        image = src.read().astype(np.float32)

    vv_values.append(image[0].flatten())
    vh_values.append(image[1].flatten())

vv = np.concatenate(vv_values)
vh = np.concatenate(vh_values)

print("\n--- VV ---")
print("Min:", np.min(vv))
print("1%:", np.percentile(vv, 1))
print("5%:", np.percentile(vv, 5))
print("25%:", np.percentile(vv, 25))
print("50%:", np.percentile(vv, 50))
print("75%:", np.percentile(vv, 75))
print("95%:", np.percentile(vv, 95))
print("99%:", np.percentile(vv, 99))
print("Max:", np.max(vv))

print("\n--- VH ---")
print("Min:", np.min(vh))
print("1%:", np.percentile(vh, 1))
print("5%:", np.percentile(vh, 5))
print("25%:", np.percentile(vh, 25))
print("50%:", np.percentile(vh, 50))
print("75%:", np.percentile(vh, 75))
print("95%:", np.percentile(vh, 95))
print("99%:", np.percentile(vh, 99))
print("Max:", np.max(vh))