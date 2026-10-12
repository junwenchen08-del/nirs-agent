"""Explicit public Corn mapping, fixed split and no guessed units/targets."""

import numpy as np
import pytest

from scripts.prepare_nir_corn import corn_arrays, prepare_corn


def source():
    rng = np.random.default_rng(29)
    return {
        "m5spec": {"data": rng.normal(size=(80, 700)), "axisscale": [[None, None], [np.arange(1100, 2500, 2), None]]},
        "propvals": {"data": rng.normal(size=(80, 4)), "label": [[None, None], [np.array(["Moisture", "Oil", "Protein", "Starch"]), None]]},
    }


def test_target_mapping_and_wavelength_axis_are_explicit():
    raw = source()
    data, metadata = corn_arrays(raw, "Oil")
    np.testing.assert_array_equal(data["y"], raw["propvals"]["data"][:, 1])
    assert data["X"].shape == (80, 700) and data["wv"].tolist() == list(range(1100, 2500, 2))
    assert metadata["target_unit"] == "not_specified_by_source"
    assert metadata["instrument"] == "m5spec" and metadata["sample_grouping"] == "one_instrument_one_row_per_sample; source_groups_not_supplied"
    with pytest.raises(ValueError, match="target"):
        corn_arrays(raw, "NotOil")


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "axis", "targets", "orientation"])
def test_invalid_mapping_and_samples_are_detected(invalid):
    raw = source()
    if invalid == "missing":
        raw["propvals"]["data"][1, 1] = np.nan
    if invalid == "duplicate":
        raw["m5spec"]["data"][1] = raw["m5spec"]["data"][0]
    if invalid == "axis":
        raw["m5spec"]["axisscale"][1][0][2] = 1100
    if invalid == "targets":
        raw["propvals"]["label"][1][0] = np.array(["Oil"] * 4)
    if invalid == "orientation":
        raw["m5spec"]["data"] = raw["m5spec"]["data"].T
    with pytest.raises(ValueError):
        corn_arrays(raw, "Oil")


def test_original_test_locked_once_and_manifest_hash_bound(tmp_path, monkeypatch):
    from scripts import prepare_nir_corn

    raw = source()
    source_path = tmp_path / "original.mat"
    source_path.write_bytes(b"unit-fixture-not-real-dataset")
    monkeypatch.setattr(prepare_nir_corn, "loadmat", lambda *args, **kw: raw)
    out = tmp_path / "corn.npz"
    manifest = prepare_corn(source_path, out)
    assert manifest["counts"] == {"calibration": 48, "validation": 16, "final_test": 16}
    assert manifest["split"]["random_state"] == 42
    with np.load(out, allow_pickle=False) as data:
        assert data["X_cal"].shape == (48, 700)
        for name, ids in manifest["partition_row_indices"].items():
            np.testing.assert_array_equal(data["y_" + name], raw["propvals"]["data"][ids, 1])
    with pytest.raises(FileExistsError):
        prepare_corn(source_path, out)
    assert source_path.read_bytes() == b"unit-fixture-not-real-dataset"
