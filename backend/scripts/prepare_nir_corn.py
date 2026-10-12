"""Explicit adapter for the verified public Corn/m5 dataset, without training.

Reuses the project's scientific gates and split_dataset. The manifest preserves
one immutable interpretation/split; it is private audit evidence, not LLM input.
This public, historically used dataset is for functionality validation only.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from nir_core.model.evaluation import split_dataset
from nir_core.utils.scientific_validation import validate_calibration_dataset, validate_partition_separation
from scipy.io import loadmat


def corn_arrays(raw, target):
    spectra, properties = raw["m5spec"], raw["propvals"]
    labels = [str(value).strip() for value in properties["label"][1][0]]
    if labels.count(target) != 1:
        raise ValueError("target must name exactly one source property label")
    X = np.asarray(spectra["data"], dtype=float)
    Y = np.asarray(properties["data"], dtype=float)
    wv = np.asarray(spectra["axisscale"][1][0], dtype=float).ravel()
    if X.shape != (80, 700) or Y.shape != (80, len(labels)) or len(wv) != 700:
        raise ValueError("Public Corn dimensions/orientation do not match the explicit mapping")
    y = Y[:, labels.index(target)]
    gate = validate_calibration_dataset(X, y, wv, minimum_samples=6)
    if not gate["passed"] or len(np.unique(X, axis=0)) != len(X):
        raise ValueError("Corn sample, target or wavelength integrity failed; no silent correction")
    return {"X": X, "y": y, "wv": wv}, {
        "dataset": "public_corn",
        "instrument": "m5spec",
        "source_sample_count": 80,
        "target": target,
        "target_column_zero_based": labels.index(target),
        "target_unit": "not_specified_by_source",
        "wavelength_unit": "nm",
        "wavelength_source": "m5spec.axisscale[1][0] and original information field",
        "sample_grouping": "one_instrument_one_row_per_sample; source_groups_not_supplied",
        "test_history": "Same Corn spectra were previously used for legacy moisture modeling; historical Test membership is unverified. Functionality validation, not unseen confirmation.",
        "source_url": "https://eigenvector.com/resources/data-sets/",
        "protocol": "sealed_holdout",
        "decision_strategy": "llm_autonomous",
        "scientific_integrity": gate,
    }


def prepare_corn(source, output, *, target="Oil", expected_source_sha256=None):
    source, output = Path(source), Path(output)
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Prepared dataset/manifest already exists; never repartition or overwrite")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected_source_sha256 and digest != expected_source_sha256:
        raise ValueError("Original source SHA-256 differs from the reviewed dataset")
    data, metadata = corn_arrays(loadmat(source, simplify_cells=True), target)
    # The splitter is unchanged and target independent. Do not select a seed
    # using model results or reuse Test as an optimizer input.
    parts = split_dataset(data["X"], np.arange(len(data["X"])), test_ratio=0.2, val_ratio=0.2, random_state=42)
    arrays = {"wv": data["wv"], "wv_unit": np.array("nm")}
    ids = {}
    for name, (_, rows) in zip(("cal", "val", "test"), parts, strict=True):
        rows = np.asarray(rows, dtype=int)
        ids[name] = rows.tolist()
        arrays["X_" + name], arrays["y_" + name] = data["X"][rows], data["y"][rows]
    separation = validate_partition_separation({name: arrays["X_" + name] for name in ids})
    if not separation["passed"]:
        raise ValueError("Prepared partitions are not scientifically separated")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file:
        np.savez(file, **arrays)
    manifest = {
        **metadata,
        "schema_version": 1,
        "source_sha256": digest,
        "prepared_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "counts": {"calibration": len(ids["cal"]), "validation": len(ids["val"]), "final_test": len(ids["test"])},
        "partition_row_indices": ids,
        "partition_separation": separation,
        "split": {"source": "nir_core.model.evaluation.split_dataset", "random_state": 42, "test_ratio": 0.2, "val_ratio": 0.2, "locked_before_modeling": True},
        "cv": {"splitter": "KFold", "n_splits": 5, "shuffle": True, "random_state": 42, "scope": "calibration_cv"},
    }
    with manifest_path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="Oil")
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()
    manifest = prepare_corn(args.source, args.output, target=args.target, expected_source_sha256=args.expected_source_sha256)
    print(json.dumps({key: manifest[key] for key in ("dataset", "target", "target_unit", "counts", "source_sha256", "prepared_sha256", "test_history")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
