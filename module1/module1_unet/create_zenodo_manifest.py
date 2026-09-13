import json
from pathlib import Path
import rasterio

oil_dir = Path(r"C:\Users\tulip\OneDrive\Desktop\SIH\datasets\Oil")
mask_dir = Path(r"C:\Users\tulip\OneDrive\Desktop\SIH\datasets\Mask_oil")

output_path = Path("module1_unet/segmentation/data/zenodo_manifest.json")

entries = []

oil_files = sorted(oil_dir.glob("*.tif"))

print(f"Oil images found: {len(oil_files)}")

for oil_file in oil_files:
    mask_file = mask_dir / oil_file.name

    if not mask_file.exists():
        print(f"WARNING: Missing mask for {oil_file.name}")
        continue

    # Read real geospatial metadata from the image
    with rasterio.open(oil_file) as src:
        crs = src.crs.to_string() if src.crs else None
        transform = list(src.transform)[:6] if src.transform else None

    entries.append({
        "scene_id": oil_file.stem,
        "patch_path": str(oil_file.resolve()).replace("\\", "/"),
        "mask_path": str(mask_file.resolve()).replace("\\", "/"),
        "acquisition_timestamp_utc": None,  # Placeholder for timestamp if available
        "crs": crs,
        "transform": transform,
        "bands": ["VV", "VH"]
    })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(entries, f, indent=2)

print(f"Paired entries created: {len(entries)}")
print(f"Manifest saved to: {output_path}")