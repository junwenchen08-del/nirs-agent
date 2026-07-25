"""Deterministic, leakage-aware data partitioning for NIR modeling tools."""

from __future__ import annotations

import numpy as np

_GROUP_FIELD_PATTERNS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (100, ("instrument", "spectrometer", "device", "仪器", "设备")),
    (95, ("batch", "lot", "批次", "批号")),
    (90, ("season", "year", "季节", "年份")),
    (85, ("region", "origin", "site", "farm", "orchard", "产地", "地区", "站点", "农场", "果园")),
    (80, ("cultivar", "variety", "population", "品种", "种类", "群体")),
)


def split_counts(n_samples: int, *, tuning_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    """Validate ratios and return exact calibration/tuning/test counts."""
    if n_samples < 6:
        raise ValueError("At least 6 samples are required for a three-way split")
    if not (0.0 < tuning_ratio < 1.0) or not (0.0 < test_ratio < 1.0):
        raise ValueError("tuning_ratio and test_ratio must both be in (0, 1)")
    if tuning_ratio + test_ratio >= 1.0:
        raise ValueError("tuning_ratio + test_ratio must be less than 1")
    n_tuning = max(1, int(round(n_samples * tuning_ratio)))
    n_test = max(1, int(round(n_samples * test_ratio)))
    n_calibration = n_samples - n_tuning - n_test
    if n_calibration < 3:
        raise ValueError("The requested split leaves fewer than 3 calibration samples")
    return n_calibration, n_tuning, n_test


def allocate_ordered_indices(order: np.ndarray, *, n_calibration: int, n_tuning: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distribute a representative sample order with weighted round-robin."""
    targets = np.asarray([n_calibration, n_tuning, n_test], dtype=int)
    current = np.zeros(3, dtype=int)
    buckets: list[list[int]] = [[], [], []]
    n_total = int(targets.sum())
    if len(order) != n_total:
        raise ValueError(f"Split order has {len(order)} rows, expected {n_total}")
    for position, raw_index in enumerate(np.asarray(order, dtype=int)):
        deficits = targets * (float(position + 1) / n_total) - current
        deficits[current >= targets] = -np.inf
        destination = int(np.argmax(deficits))
        buckets[destination].append(int(raw_index))
        current[destination] += 1
    return tuple(np.sort(np.asarray(bucket, dtype=int)) for bucket in buckets)  # type: ignore[return-value]


def deterministic_holdout_indices(n_samples: int, *, tuning_ratio: float, test_ratio: float, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = split_counts(n_samples, tuning_ratio=tuning_ratio, test_ratio=test_ratio)
    order = np.random.RandomState(random_state).permutation(n_samples)
    return allocate_ordered_indices(order, n_calibration=counts[0], n_tuning=counts[1], n_test=counts[2])


def y_stratified_holdout_indices(y: np.ndarray, *, tuning_ratio: float, test_ratio: float, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).ravel()
    counts = split_counts(len(y), tuning_ratio=tuning_ratio, test_ratio=test_ratio)
    sorted_indices = np.argsort(y, kind="mergesort")
    rng = np.random.RandomState(random_state)
    blocks: list[np.ndarray] = []
    for start in range(0, len(y), 20):
        block = sorted_indices[start : start + 20].copy()
        rng.shuffle(block)
        blocks.append(block)
    return allocate_ordered_indices(
        np.concatenate(blocks),
        n_calibration=counts[0],
        n_tuning=counts[1],
        n_test=counts[2],
    )


def spxy_holdout_indices(X: np.ndarray, y: np.ndarray, *, tuning_ratio: float, test_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a joint spectral/target maximin order and distribute it evenly."""
    from sklearn.metrics import pairwise_distances

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    counts = split_counts(len(y), tuning_ratio=tuning_ratio, test_ratio=test_ratio)
    feature_std = np.std(X, axis=0, ddof=0)
    feature_std[feature_std < 1e-12] = 1.0
    spectral_distance = pairwise_distances((X - np.mean(X, axis=0)) / feature_std, metric="euclidean")
    spectral_max = float(np.max(spectral_distance))
    if spectral_max > 0:
        spectral_distance /= spectral_max
    y_distance = np.abs(y[:, None] - y[None, :])
    y_max = float(np.max(y_distance))
    if y_max > 0:
        y_distance /= y_max
    joint_distance = spectral_distance + y_distance
    np.fill_diagonal(joint_distance, -np.inf)
    first, second = np.unravel_index(int(np.argmax(joint_distance)), joint_distance.shape)
    selected = np.zeros(len(y), dtype=bool)
    selected[[first, second]] = True
    order = [int(first), int(second)]
    min_distance = np.minimum(joint_distance[:, first], joint_distance[:, second])
    min_distance[selected] = -np.inf
    while len(order) < len(y):
        next_index = int(np.argmax(min_distance))
        if selected[next_index]:
            next_index = int(np.flatnonzero(~selected)[0])
        order.append(next_index)
        selected[next_index] = True
        min_distance = np.minimum(min_distance, joint_distance[:, next_index])
        min_distance[selected] = -np.inf
    return allocate_ordered_indices(
        np.asarray(order, dtype=int),
        n_calibration=counts[0],
        n_tuning=counts[1],
        n_test=counts[2],
    )


def group_field_candidates(raw, *, y_col: int) -> list[dict]:
    """Rank plausible batch/domain grouping columns and reject ID-like fields."""
    max_groups = max(5, min(100, int(round(len(raw) * 0.25))))
    candidates: list[dict] = []
    for column_index, column in enumerate(raw.columns):
        if column_index == int(y_col):
            continue
        name = str(column)
        normalized = "".join(character.lower() for character in name if character.isalnum() or "\u4e00" <= character <= "\u9fff")
        if any(token in normalized for token in ("sampleid", "recordid", "样本编号", "样本id")) or normalized in {"id", "index", "序号"}:
            continue
        score = next(
            (weight for weight, patterns in _GROUP_FIELD_PATTERNS if any(pattern in normalized for pattern in patterns)),
            None,
        )
        if score is None:
            continue
        values = raw[column]
        missing_fraction = float(values.isna().mean())
        n_groups = int(values.nunique(dropna=True))
        candidates.append(
            {
                "column": name,
                "score": int(score),
                "n_groups": n_groups,
                "missing_fraction": missing_fraction,
                "eligible": missing_fraction <= 0.20 and 5 <= n_groups <= max_groups,
            }
        )
    return sorted(
        candidates,
        key=lambda item: (
            -item["eligible"],
            -item["score"],
            item["n_groups"],
            item["column"],
        ),
    )


def _choose_group_subset(
    values: np.ndarray,
    available_groups: list[str],
    *,
    target_samples: int,
    rng,
    min_groups_left: int,
) -> set[str]:
    shuffled = list(available_groups)
    rng.shuffle(shuffled)
    counts = {group: int(np.sum(values == group)) for group in shuffled}
    selected: set[str] = set()
    selected_count = 0
    while len(shuffled) - len(selected) > min_groups_left:
        remaining = [group for group in shuffled if group not in selected]
        best = min(remaining, key=lambda group: abs(selected_count + counts[group] - target_samples))
        if selected and abs(selected_count + counts[best] - target_samples) >= abs(selected_count - target_samples):
            break
        selected.add(best)
        selected_count += counts[best]
    if not selected:
        selected.add(min(shuffled, key=lambda group: abs(counts[group] - target_samples)))
    return selected


def group_holdout_indices(values, *, tuning_ratio: float, test_ratio: float, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    group_values = np.asarray(
        ["<missing>" if value is None or str(value) == "nan" else str(value) for value in values],
        dtype=object,
    )
    groups = sorted(set(group_values.tolist()))
    if len(groups) < 3:
        raise ValueError("Group splitting requires at least three distinct groups")
    _, n_tuning, n_test = split_counts(len(group_values), tuning_ratio=tuning_ratio, test_ratio=test_ratio)
    rng = np.random.RandomState(random_state)
    test_groups = _choose_group_subset(group_values, groups, target_samples=n_test, rng=rng, min_groups_left=2)
    remaining = [group for group in groups if group not in test_groups]
    tuning_groups = _choose_group_subset(group_values, remaining, target_samples=n_tuning, rng=rng, min_groups_left=1)
    calibration_groups = set(remaining) - tuning_groups
    calibration = np.flatnonzero(np.isin(group_values, list(calibration_groups)))
    tuning = np.flatnonzero(np.isin(group_values, list(tuning_groups)))
    holdout = np.flatnonzero(np.isin(group_values, list(test_groups)))
    if min(len(calibration), len(tuning), len(holdout)) == 0:
        raise ValueError("Group splitting produced an empty partition")
    return (
        np.sort(calibration),
        np.sort(tuning),
        np.sort(holdout),
        {
            "calibration": sorted(calibration_groups),
            "tuning": sorted(tuning_groups),
            "holdout_test": sorted(test_groups),
        },
    )


def select_autonomous_split(
    raw,
    X: np.ndarray,
    y: np.ndarray,
    *,
    y_col: int,
    strategy: str,
    group_col: str | None,
    tuning_ratio: float,
    test_ratio: float,
    random_state: int,
    spxy_max_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Select a deterministic split using grouping fields and sample size."""
    requested = (strategy or "auto").strip().lower().replace("-", "_")
    requested = {"stratified": "y_stratified", "stratified_y": "y_stratified", "ks": "spxy"}.get(requested, requested)
    allowed = {"auto", "group", "spxy", "y_stratified", "random"}
    if requested not in allowed:
        raise ValueError(f"Unknown split_strategy {strategy!r}; use one of: {', '.join(sorted(allowed))}")
    if int(spxy_max_samples) < 6:
        raise ValueError("spxy_max_samples must be at least 6")
    candidates = group_field_candidates(raw, y_col=y_col)
    selected_group = group_col
    if selected_group is not None and selected_group not in raw.columns:
        raise ValueError(f"group_col {selected_group!r} is not a CSV header")
    effective = requested
    if requested == "auto":
        selected_group = selected_group or next((item["column"] for item in candidates if item["eligible"]), None)
        if selected_group is not None:
            effective = "group"
            reason = f"Detected grouping field {selected_group!r}; whole groups are isolated to prevent batch/domain leakage."
        elif len(y) <= int(spxy_max_samples):
            effective = "spxy"
            reason = f"No eligible grouping field; n_samples={len(y)} is within the SPXY limit {int(spxy_max_samples)}."
        else:
            effective = "y_stratified"
            reason = f"No eligible grouping field; n_samples={len(y)} exceeds the SPXY limit {int(spxy_max_samples)}, so quadratic distances are avoided."
    elif requested == "group":
        selected_group = selected_group or next((item["column"] for item in candidates if item["eligible"]), None)
        if selected_group is None:
            raise ValueError("split_strategy='group' requires group_col or an eligible detected grouping field")
        reason = f"Group split explicitly requested with field {selected_group!r}."
    elif requested == "spxy":
        if len(y) > int(spxy_max_samples):
            raise ValueError(f"SPXY requested for {len(y)} samples, above spxy_max_samples={int(spxy_max_samples)}")
        reason = "SPXY explicitly requested; joint spectral and target distances determine a balanced maximin order."
    elif requested == "y_stratified":
        reason = "Target-value stratification explicitly requested."
    else:
        reason = "Deterministic random split explicitly requested."

    group_assignments = None
    if effective == "group":
        calibration, tuning, holdout, group_assignments = group_holdout_indices(
            raw[selected_group].to_numpy(),
            tuning_ratio=tuning_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
        )
    elif effective == "spxy":
        calibration, tuning, holdout = spxy_holdout_indices(X, y, tuning_ratio=tuning_ratio, test_ratio=test_ratio)
    elif effective == "y_stratified":
        calibration, tuning, holdout = y_stratified_holdout_indices(
            y,
            tuning_ratio=tuning_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
        )
    else:
        calibration, tuning, holdout = deterministic_holdout_indices(
            len(y),
            tuning_ratio=tuning_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
        )
    n_total = len(y)
    return (
        calibration,
        tuning,
        holdout,
        {
            "requested_strategy": requested,
            "strategy": effective,
            "reason": reason,
            "random_state": int(random_state),
            "spxy_max_samples": int(spxy_max_samples),
            "group_column": selected_group if effective == "group" else None,
            "group_field_candidates": candidates,
            "group_assignments": group_assignments,
            "actual_ratios": {
                "calibration": float(len(calibration) / n_total),
                "tuning": float(len(tuning) / n_total),
                "holdout_test": float(len(holdout) / n_total),
            },
        },
    )
