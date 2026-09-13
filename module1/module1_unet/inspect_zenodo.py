import rasterio
import numpy as np

image_path = r"C:\Users\tulip\OneDrive\Desktop\SIH\datasets\Oil\00001.tif"
mask_path = r"C:\Users\tulip\OneDrive\Desktop\SIH\datasets\Mask_oil\00001.tif"
print("\n--- IMAGE ---")

with rasterio.open(image_path) as src:
    image = src.read()

    print("Shape:", image.shape)
    print("Dtype:", image.dtype)
    print("CRS:", src.crs)
    print("Transform:", src.transform)
    print("Bands:", src.count)

    for i in range(src.count):
        band = image[i]
        print(
            f"Band {i+1}: "
            f"min={np.nanmin(band):.4f}, "
            f"max={np.nanmax(band):.4f}, "
            f"mean={np.nanmean(band):.4f}"
        )

print("\n--- MASK ---")

with rasterio.open(mask_path) as src:
    mask = src.read(1)

    print("Shape:", mask.shape)
    print("Dtype:", mask.dtype)
    print("Unique values:", np.unique(mask))
    print("CRS:", src.crs)