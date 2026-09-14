import numpy as np
import rasterio
from affine import Affine

transform = Affine(0.0001, 0, 80.25, 0, -0.0001, 13.15)
crs = "EPSG:4326"
size = 512

# Create simulated linear Sigma0 backscatter values (0.001 to 0.1)
vv_data = np.random.uniform(0.01, 0.08, (size, size)).astype(np.float32)
vh_data = np.random.uniform(0.002, 0.02, (size, size)).astype(np.float32)

profile = {
    "driver": "GTiff",
    "height": size,
    "width": size,
    "count": 1,
    "dtype": rasterio.float32,
    "crs": crs,
    "transform": transform
}

with rasterio.open("test_vv.tif", "w", **profile) as dst:
    dst.write(vv_data, 1)

with rasterio.open("test_vh.tif", "w", **profile) as dst:
    dst.write(vh_data, 1)

print("Created test_vv.tif and test_vh.tif")