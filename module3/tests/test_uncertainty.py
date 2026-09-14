from src.uncertainty import fixed_radius_heuristic


def test_uncertainty_scaling():
    assert fixed_radius_heuristic(5, 0.5, 0) == 5
    assert fixed_radius_heuristic(5, 0.5, 10) == 10
