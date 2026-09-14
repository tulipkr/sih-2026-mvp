"""
Real Sentinel-1 GRD measurement GeoTIFFs are usually NOT map-projected —
`rasterio` will report `src.crs is None` and `src.transform` as the identity
matrix, because the product instead carries Ground Control Points (GCPs)
tying pixel positions to real-world coordinates. This is a genuine,
well-known property of the raw product (not a bug in this pipeline), and it
matters here specifically: Module 2 spec §19 requires a correct affine
transform, and §4/§38 forbid fabricating one.

`resolve_geotransform()` derives a best-effort affine transform from the
product's own GCPs when no direct transform is present — an approximation
(a single best-fit affine over the whole scene), not true GCP-based
orthorectification/warping. This is flagged explicitly wherever it's used;
it is adequate for tiling and approximate georeferencing at the coarse
resolution this MVP operates at, but is NOT a substitute for a real
`gdalwarp`-style GCP correction if the project's later stages need
survey-grade accuracy.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class GeoreferencingError(ValueError):
    pass


def resolve_geotransform(src):
    """
    Args:
        src: an open rasterio dataset.
    Returns:
        (transform, crs, is_gcp_approximation: bool)
    Raises:
        GeoreferencingError: no transform AND no GCPs at all — genuinely
        ungeoreferenced input, which spec §16 says must be a hard error
        ("CRS missing after calibration -> cannot proceed to tiling").
    """
    if src.crs is not None and not src.transform.is_identity:
        return src.transform, src.crs, False

    gcps, gcp_crs = src.gcps
    if not gcps:
        raise GeoreferencingError(
            f"{src.name}: no CRS/transform AND no GCPs present — this file "
            f"has no usable georeferencing at all. Cannot proceed to tiling "
            f"(spec §16)."
        )

    import rasterio.transform

    logger.warning(
        f"{src.name}: no direct CRS/transform; deriving an approximate affine "
        f"transform from {len(gcps)} GCPs. This is a whole-scene best-fit "
        f"approximation, not a true GCP warp — acceptable for this MVP's "
        f"tiling/patch georeferencing, flagged here so it's never mistaken "
        f"for a fully orthorectified product."
    )
    approx_transform = rasterio.transform.from_gcps(gcps)
    return approx_transform, gcp_crs, True
