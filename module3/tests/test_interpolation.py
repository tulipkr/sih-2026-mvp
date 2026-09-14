import numpy as np
import xarray as xr
from src.interpolation import bilinear_or_nearest_valid


def test_interpolation_known_grid():
    ds = xr.Dataset(
        {"u10": (("latitude", "longitude"), np.array([[0., 1.], [2., 3.]]))},
        coords={"latitude": [0., 1.], "longitude": [0., 1.]},
    )
    value, used_nearest = bilinear_or_nearest_valid(ds, "u10", 0.5, 0.5)
    assert value == 1.5
    assert used_nearest is False
