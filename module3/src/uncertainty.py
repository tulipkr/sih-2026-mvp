from __future__ import annotations

import math


def fixed_radius_heuristic(base_radius_km, growth_km_per_hour, elapsed_hours):
    return float(base_radius_km + growth_km_per_hour * elapsed_hours)


def uncertainty_radius(config, elapsed_hours):
    method = config["uncertainty_method"]
    if method == "fixed_radius_heuristic":
        return fixed_radius_heuristic(
            config["uncertainty_base_radius_km"],
            config["uncertainty_growth_km_per_hour"],
            elapsed_hours,
        )
    if method == "perturbation_ensemble":
        raise NotImplementedError("perturbation_ensemble")
    raise ValueError("uncertainty_method")
