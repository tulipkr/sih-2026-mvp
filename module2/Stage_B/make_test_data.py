import json
import numpy as np
import rasterio
from affine import Affine

mask = np.zeros((512, 512), dtype=np.uint8)
mask[200:260, 200:270] = 1  # Spill 1
mask[100:120, 100:130] = 1  # Spill 2
mask[10:11, 10:11] = 1      # Noise speck to be discarded

transform = Affine(0.0001, 0, 80.3, 0, -0.0001, 13.1)
crs = "EPSG:4326"

with rasterio.open(
    "test_mask.tif", "w", driver="GTiff",
    height=512, width=512, count=1,
    dtype=rasterio.uint8, crs=crs, transform=transform
) as dst:
    dst.write(mask, 1)

meta = {
    "scene_id": "S1A_CHENNAI_DEMO",
    "crs": crs,
    "transform": list(transform)[:6],
    "acquisition_timestamp_utc": "2023-12-04T00:30:00Z",
    "mask_path": "test_mask.tif",
    "no_oil_detected": False,
    "geolocation_incomplete": False,
    "mean_confidence": 0.95
}

with open("test_inference.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Test inputs generated.")