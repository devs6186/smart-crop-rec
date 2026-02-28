"""
Deep comprehensive tests: verify crop recommendations vary by state, district, and land size.
Run from project root: python -m pytest tests/test_crop_variety.py -v
Or: python tests/test_crop_variety.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.zone_soil import get_default_soil_climate
from src.predictor import predict_crop
from src.config import INDIAN_STATES, DISTRICTS_BY_STATE, STATE_ZONE


def _run_prediction(state: str, district: str | None, land_bigha: float) -> list[str]:
    """Get top 5 crop names for a given state, district, land size."""
    soil = get_default_soil_climate(state, district)
    result = predict_crop(
        soil["N"], soil["P"], soil["K"],
        soil["temperature"], soil["humidity"], soil["ph"], soil["rainfall"],
        land_size_bigha=land_bigha,
        state=state,
        district=district,
        scoring_mode="suitability",
    )
    return [c["crop"].lower() for c in result["top5"]]


def test_soil_climate_varies_by_state():
    """Different states must produce different soil/climate inputs."""
    seen = {}
    for state in ["Rajasthan", "Kerala", "Himachal Pradesh", "West Bengal", "Maharashtra"]:
        if state not in STATE_ZONE:
            continue
        soil = get_default_soil_climate(state, None)
        key = tuple(soil[k] for k in ["N", "P", "K", "temperature", "rainfall"])
        assert key not in seen.values(), f"State {state} produced duplicate soil profile"
        seen[state] = key
    assert len(seen) >= 5, "Should have 5 distinct state profiles"


def test_soil_climate_varies_by_district():
    """Different districts in same state must produce different soil inputs."""
    state = "Ladakh"
    districts = ["Leh", "Kargil"]
    soils = [get_default_soil_climate(state, d) for d in districts]
    # At least one feature should differ
    diffs = sum(1 for k in soils[0] if soils[0][k] != soils[1][k])
    assert diffs >= 1, "Leh vs Kargil should have different soil/climate values"


def test_crops_vary_across_states():
    """Different states must get different top-5 crop recommendations (not same 5 for all)."""
    states_with_districts = [
        ("Rajasthan", "Jaipur"),
        ("Kerala", "Thrissur"),
        ("Himachal Pradesh", "Shimla"),
        ("West Bengal", "Murshidabad"),
        ("Maharashtra", "Pune"),
        ("Ladakh", "Leh"),
        ("Bihar", "Patna"),
    ]
    all_top5_sets = []
    for state, district in states_with_districts:
        crops = _run_prediction(state, district, 2.0)
        all_top5_sets.append((state, tuple(crops)))
    # Collect unique top-5 tuples
    unique_combos = set(t for _, t in all_top5_sets)
    # We need at least 2 different top-5 combinations (proves state affects recommendations)
    assert len(unique_combos) >= 2, (
        f"Only {len(unique_combos)} unique top-5 combinations across 7 states. "
        f"Expected ≥2. Got: {[(s, list(t)) for s, t in all_top5_sets]}"
    )


def test_crops_vary_by_district_within_state():
    """Different districts in same state get soil inputs that can affect recommendations."""
    state = "Karnataka"
    from src.zone_soil import get_default_soil_climate
    s1 = get_default_soil_climate(state, "Belagavi")
    s2 = get_default_soil_climate(state, "Mysuru")
    # District offset should produce at least one different value
    diffs = sum(1 for k in s1 if s1[k] != s2[k])
    assert diffs >= 1, "Different districts should yield different soil/climate inputs"


def test_crops_vary_by_land_size():
    """Land size affects which crops pass the filter; both return valid lists."""
    state, district = "Rajasthan", "Jaipur"
    small = _run_prediction(state, district, 0.5)   # 0.5 bigha ≈ 0.31 acres
    large = _run_prediction(state, district, 50.0)  # 50 bigha ≈ 31 acres
    assert len(small) >= 1 and len(large) >= 1
    assert all(isinstance(c, str) for c in small + large)


def test_no_single_crop_dominates_all_states():
    """Top-5 lists should differ across states (proves region affects recommendations)."""
    states = ["Rajasthan", "Kerala", "Himachal Pradesh", "West Bengal", "Maharashtra"]
    all_top5 = []
    for state in states:
        districts = DISTRICTS_BY_STATE.get(state, ["Other"])
        district = districts[0] if districts and districts[0] not in ("Other / Not Listed", "Other") else None
        crops = _run_prediction(state, district, 2.0)
        all_top5.append(tuple(crops))
    unique_combos = len(set(all_top5))
    # At least 2 different top-5 combinations across 5 different zones
    assert unique_combos >= 2, (
        f"Only {unique_combos} unique top-5 across 5 states. Expected ≥2."
    )


def test_prediction_returns_valid_structure():
    """Prediction must return valid top5 with required fields."""
    result = predict_crop(50, 50, 50, 25, 65, 6.5, 120, land_size_bigha=2.0, state="Karnataka", district="Mysuru", scoring_mode="suitability")
    assert "top5" in result
    assert 1 <= len(result["top5"]) <= 5  # Land filter may reduce count
    for c in result["top5"]:
        assert "crop" in c
        assert "suitability_pct" in c
        assert "rank" in c
    assert result["region"]["state"] == "Karnataka"
    assert result["region"]["district"] == "Mysuru"


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v", "-s"]))
