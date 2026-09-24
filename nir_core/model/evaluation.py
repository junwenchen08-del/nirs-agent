"""Evaluation utilities: dataset splitting, cross-validation, nested-CV
preprocessing selection, and automatic component-count selection.

The metric functions are NOT re-implemented here; ``compute_metrics`` is
re-exported from :mod:`nir_core.utils.metrics` for convenience.

Key design rules enforced by this module:

- **No leakage**: ``split_dataset`` produces three mutually disjoint sets;
  ``cross_validate`` rebuilds the model on each training fold only;
  ``nested_cv_preprocessing`` never lets the validation set touch the
  inner CV component search, and each inner fold *refits* the preprocessing
  pipeline on the inner-training subset so that stateful steps (mean-center,
  autoscale, MSC) cannot leak inner-validation statistics into the
  inner-training representation.
- **Reproducibility**: every RNG-using function accepts ``random_state``.
- **Numerical safety**: component counts are clamped to
  ``min(n_samples-1, n_wavelengths)``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, train_test_split

from nir_core.preprocess.pipeline import (
    DEFAULT_CANDIDATE_PIPELINES,
    PreprocessingPipeline,
)
from nir_core.utils.metrics import compute_metrics, r2_score, rmse

__all__ = [
    "auto_select_components",
    "compute_metrics",
    "cross_validate",
    "nested_cv_preprocessing",
    "split_dataset",
]


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.20,
    val_ratio: float = 0.10,
    random_state: int = 42,
) -> tuple[tuple, tuple, tuple]:
    """Split data into disjoint train / validation / test sets.

    Strategy: first peel off the test set from the full data; then peel
    off the validation set from the remaining (train+val) pool. The
    ``val_ratio`` is interpreted *relative to the original dataset size*
    (not the post-test remainder), so the final proportions approximate
    ``test_ratio`` and ``val_ratio`` of the full data.

    Args:
        X: Spectra, shape (n_samples, n_wavelengths).
        y: Reference values, shape (n_samples,) or (n_samples, n_targets).
        test_ratio: Fraction of the full data reserved as test
            (0 < test_ratio < 1).
        val_ratio: Fraction of the full data reserved as validation
            (0 < val_ratio < 1, and ``test_ratio + val_ratio < 1``).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of three ``((X_train, y_train), (X_val, y_val),
        (X_test, y_test))``. The three sets are guaranteed to have no
        overlapping row indices.

    Raises:
        ValueError: On shape mismatch or invalid ratios.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim not in {1, 2}:
        raise ValueError(f"y must be 1D or 2D, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X rows ({X.shape[0]}) != y length ({y.shape[0]})")
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in (0,1), got {test_ratio}")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")
    if test_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"test_ratio + val_ratio ({test_ratio + val_ratio}) must be < 1"
        )

    X_rem, X_test, y_rem, y_test = train_test_split(
        X,
        y,
        test_size=test_ratio,
        random_state=random_state,
        shuffle=True,
    )

    n_total = X.shape[0]
    n_rem = X_rem.shape[0]
    val_frac_of_rem = (val_ratio * n_total) / n_rem
    val_frac_of_rem = min(max(val_frac_of_rem, 1e-6), 1.0 - 1e-6)

    X_train, X_val, y_train, y_val = train_test_split(
        X_rem,
        y_rem,
        test_size=val_frac_of_rem,
        random_state=random_state,
        shuffle=True,
    )

    return ((X_train, y_train), (X_val, y_val), (X_test, y_test))


def cross_validate(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 10,
    random_state: int = 42,
) -> dict:
    """K-fold cross-validation for any model exposing ``fit``/``predict``.

    A *fresh* model is built via ``model_factory()`` on each training fold,
    so no state leaks between folds.

    Args:
        model_factory: Zero-argument callable returning a fresh, un-fitted
            model object with ``.fit(X, y)`` and ``.predict(X)`` methods.
            For 2D ``y`` the returned model must support multi-output
            (e.g. ``PLSRegression``, ``LinearRegression``, ``Ridge``);
            single-output estimators (e.g. ``SVR``) must be wrapped in
            ``MultiOutputRegressor`` by the caller.
        X: Spectra, shape (n_samples, n_wavelengths).
        y: Reference values, shape (n_samples,) or (n_samples, n_targets).
            When 2D, metrics are computed per-target then averaged across
            targets per fold; per-target detail is returned in
            ``per_target``. The model returned by ``model_factory`` must
            support multi-output.
        n_folds: Number of K-fold splits. Clamped to ``[2, n_samples-1]``.
        random_state: Seed for KFold shuffling.

    Returns:
        Dict with keys ``fold_rmse``, ``mean_rmse``, ``std_rmse``,
        ``fold_r2``, ``mean_r2``, ``n_folds``. For 2D ``y``,
        ``fold_rmse``/``fold_r2`` hold the per-fold mean across targets
        (still ``list[float]`` for backward compatibility), and extra keys
        ``n_targets`` and ``per_target`` are added: ``per_target`` is a
        list of per-target dicts with ``target_index``, ``fold_rmse``,
        ``fold_r2``, ``mean_rmse``, ``mean_r2``.

    Raises:
        ValueError: On shape mismatch or invalid y dimensions.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim not in (1, 2):
        raise ValueError(f"y must be 1D or 2D, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X rows ({X.shape[0]}) != y length ({y.shape[0]})")
    n_samples = X.shape[0]
    is_multi = y.ndim == 2
    n_targets = y.shape[1] if is_multi else 1
    eff_folds = max(2, min(int(n_folds), n_samples - 1))
    kf = KFold(n_splits=eff_folds, shuffle=True, random_state=random_state)

    fold_rmse: list[float] = []
    fold_r2: list[float] = []
    # Per-target fold series. For 1D y this is a single-target list whose
    # contents mirror fold_rmse/fold_r2; for 2D y each target gets its own
    # series so callers can inspect per-component CV behaviour without the
    # cross-target mixing that rmse()/r2_score() would introduce by raveling.
    per_target: list[dict] = [
        {"fold_rmse": [], "fold_r2": []} for _ in range(n_targets)
    ]

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = model_factory()
        model.fit(X_tr, y_tr)
        pred = np.asarray(model.predict(X_val))

        if is_multi:
            fold_rmses_t: list[float] = []
            fold_r2s_t: list[float] = []
            for t in range(n_targets):
                y_val_t = y_val[:, t].ravel()
                pred_t = pred[:, t].ravel()
                rmse_t = rmse(y_val_t, pred_t)
                r2_t = r2_score(y_val_t, pred_t)
                per_target[t]["fold_rmse"].append(rmse_t)
                per_target[t]["fold_r2"].append(r2_t)
                fold_rmses_t.append(rmse_t)
                fold_r2s_t.append(r2_t)
            # Top-level fold metric = mean across targets, keeps
            # fold_rmse/fold_r2 as list[float] for backward compatibility.
            fold_rmse.append(float(np.mean(fold_rmses_t)))
            fold_r2.append(float(np.mean(fold_r2s_t)))
        else:
            pred = pred.ravel()
            rmse_t = rmse(y_val, pred)
            r2_t = r2_score(y_val, pred)
            fold_rmse.append(rmse_t)
            fold_r2.append(r2_t)
            per_target[0]["fold_rmse"].append(rmse_t)
            per_target[0]["fold_r2"].append(r2_t)

    result: dict = {
        "fold_rmse": fold_rmse,
        "mean_rmse": float(np.mean(fold_rmse)),
        "std_rmse": float(np.std(fold_rmse, ddof=1)) if len(fold_rmse) > 1 else 0.0,
        "fold_r2": fold_r2,
        "mean_r2": float(np.mean(fold_r2)),
        "n_folds": eff_folds,
    }
    if is_multi:
        result["n_targets"] = n_targets
        result["per_target"] = [
            {
                "target_index": t,
                "fold_rmse": per_target[t]["fold_rmse"],
                "fold_r2": per_target[t]["fold_r2"],
                "mean_rmse": float(np.mean(per_target[t]["fold_rmse"])),
                "mean_r2": float(np.mean(per_target[t]["fold_r2"])),
            }
            for t in range(n_targets)
        ]
    return result


def _pls_cv_best_components(
    pipeline: PreprocessingPipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_folds: int,
    max_components: int,
    random_state: int,
    wv: np.ndarray | None = None,
    primary_metric: str = "r2",
) -> tuple[int, float, float]:
    """Run inner K-fold CV to pick the PLS component count on *train only*.

    **Leakage-safe**: for each inner fold the preprocessing pipeline is
    *refit* on the inner-training subset only, then used to transform both
    the inner-training and inner-validation subsets. This ensures stateful
    steps (mean-center, autoscale, MSC) never see inner-validation data.

    Args:
        pipeline: Preprocessing pipeline template (its ``steps`` are cloned
            per fold; the passed object is not mutated).
        X_train: Training spectra (raw, un-preprocessed).
        y_train: Training references.
        inner_folds: Number of K-fold splits.
        max_components: Upper bound for the component search.
        random_state: Seed for inner KFold.
        wv: Optional wavelength axis forwarded to ``detrend``.

    Returns:
        ``(best_n_comp, mean_cv_r2, mean_cv_rmse)`` at the selected component
        count. ``primary_metric`` controls whether the component count is
        selected by maximum R² (legacy) or minimum RMSE.
    """
    n_samples, n_wavelengths = X_train.shape
    upper = max(1, min(int(max_components), n_samples - 1, n_wavelengths))
    eff_folds = max(2, min(int(inner_folds), n_samples - 1))
    kf = KFold(n_splits=eff_folds, shuffle=True, random_state=random_state)

    n_comp_list = list(range(1, upper + 1))
    mean_r2_per_nc: list[float] = []
    mean_rmse_per_nc: list[float] = []

    for nc in n_comp_list:
        fold_r2: list[float] = []
        fold_rmse: list[float] = []
        for tr_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            if X_tr.shape[0] <= nc:
                continue
            try:
                # Refit pipeline on inner-train ONLY, then transform both
                # subsets with the inner-train statistics.
                p = pipeline.unfitted_copy().fit(X_tr, wv)
                X_tr_pp = p.transform(X_tr, wv)
                X_val_pp = p.transform(X_val, wv)
                m = PLSRegression(n_components=nc, scale=False)
                m.fit(X_tr_pp, y_tr)
                pred = m.predict(X_val_pp).ravel()
                fold_r2.append(r2_score(y_val, pred))
                fold_rmse.append(rmse(y_val, pred))
            except Exception:  # noqa: BLE001, S112 - one failed fold is isolated
                continue
        if not fold_r2:
            mean_r2_per_nc.append(float("-inf"))
            mean_rmse_per_nc.append(float("inf"))
        else:
            mean_r2_per_nc.append(float(np.mean(fold_r2)))
            mean_rmse_per_nc.append(float(np.mean(fold_rmse)))

    if primary_metric == "r2":
        best_idx = int(np.argmax(mean_r2_per_nc))
    elif primary_metric == "rmse":
        best_idx = int(np.argmin(mean_rmse_per_nc))
    else:
        raise ValueError(
            f"primary_metric must be 'r2' or 'rmse', got {primary_metric!r}."
        )
    best_nc = int(n_comp_list[best_idx])
    best_mean_r2 = float(mean_r2_per_nc[best_idx])
    best_mean_rmse = float(mean_rmse_per_nc[best_idx])
    if not np.isfinite(best_mean_r2):
        best_mean_r2 = 0.0
    return best_nc, best_mean_r2, best_mean_rmse


def _pipeline_simplicity_key(
    pipeline: PreprocessingPipeline,
    candidate_id: str,
) -> tuple[int, int, int, str]:
    """Return a deterministic simplicity ordering for near-tied candidates."""
    n_steps = len(pipeline.steps)
    n_parameters = sum(len(step.params) for step in pipeline.steps)
    n_stateful = sum(
        1
        for step in pipeline.steps
        if step.method in {"mean_center", "autoscale", "msc", "emsc"}
    )
    return n_steps, n_parameters, n_stateful, candidate_id


def _select_preprocessing_winner(
    pipelines: list[PreprocessingPipeline],
    evaluations: list[dict],
    *,
    selection_rule: str,
) -> tuple[int, dict]:
    """Select a valid candidate using a deterministic documented rule."""
    valid = [
        index
        for index, evidence in enumerate(evaluations)
        if "error" not in evidence
        and np.isfinite(float(evidence.get("cv_rmse", float("inf"))))
    ]
    if not valid:
        raise ValueError(
            "No candidate pipeline produced a valid score; "
            "check preprocessing methods or input data."
        )

    if selection_rule == "legacy_score":
        winner = max(
            valid,
            key=lambda index: float(evaluations[index].get("score", float("-inf"))),
        )
        decision = {
            "selection_rule": selection_rule,
            "reason_code": "maximum_legacy_combined_r2_score",
            "selected_candidate_id": evaluations[winner]["candidate_id"],
            "eligible_candidate_ids": [evaluations[winner]["candidate_id"]],
        }
        return winner, decision

    if selection_rule != "rmsecv_1pct":
        raise ValueError(
            "selection_rule must be 'legacy_score' or 'rmsecv_1pct'; "
            f"got {selection_rule!r}."
        )

    best_rmse = min(float(evaluations[index]["cv_rmse"]) for index in valid)
    threshold = best_rmse * 1.01
    eligible = [
        index for index in valid if float(evaluations[index]["cv_rmse"]) <= threshold
    ]
    winner = min(
        eligible,
        key=lambda index: _pipeline_simplicity_key(
            pipelines[index],
            str(evaluations[index]["candidate_id"]),
        ),
    )
    decision = {
        "selection_rule": selection_rule,
        "reason_code": "simplest_within_one_percent_rmsecv",
        "best_cv_rmse": best_rmse,
        "materiality_threshold_cv_rmse": threshold,
        "selected_candidate_id": evaluations[winner]["candidate_id"],
        "eligible_candidate_ids": [
            evaluations[index]["candidate_id"] for index in eligible
        ],
        "tie_break": "fewest_steps_then_parameters_then_stateful_then_id",
    }
    return winner, decision


def nested_cv_preprocessing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    candidate_pipelines: list[PreprocessingPipeline] | None = None,
    inner_folds: int = 5,
    model_method: str = "pls",
    max_components: int = 20,
    random_state: int = 42,
    wv: np.ndarray | None = None,
    selection_rule: str = "legacy_score",
    candidate_ids: list[str] | None = None,
) -> tuple[PreprocessingPipeline, dict]:
    """Select the best preprocessing pipeline via nested CV.

    For each candidate pipeline:

    1. **Fit** the pipeline on ``X_train`` (training statistics only) and
       **transform** both ``X_train`` and ``X_val`` with those statistics.
       This prevents MSC / mean-center / autoscale from leaking validation
       information.
    2. On the *raw training set*, run an inner K-fold CV that **refits** the
       pipeline on each inner-training fold to pick the optimal PLS
       component count and record ``cv_r2`` (mean inner-CV R^2). The
       validation set is **never** used in this inner search, and inner-fold
       statistics never leak across folds.
    3. Fit a PLS model on the full preprocessed training set using the
       best component count, predict the preprocessed validation set, and
       record ``val_r2``.
    4. Score = ``0.7 * cv_r2 + 0.3 * val_r2``.

    The pipeline with the highest score is returned.

    Args:
        X_train: Training spectra (raw).
        y_train: Training references.
        X_val: Validation spectra (raw).
        y_val: Validation references.
        candidate_pipelines: List of :class:`PreprocessingPipeline`. If
            ``None``, uses :data:`DEFAULT_CANDIDATE_PIPELINES`.
        inner_folds: Number of K-fold splits for the inner component search.
        model_method: Currently only ``"pls"`` is supported.
        max_components: Upper bound for the PLS component search.
        random_state: Seed for inner KFold.
        wv: Optional wavelength axis forwarded to ``detrend`` steps.

    Returns:
        Tuple ``(best_pipeline, results_dict)`` where ``results_dict``
        has one entry per candidate, keyed by the pipeline's
        ``description()``, each a sub-dict with keys ``score``, ``cv_r2``,
        ``val_r2``, ``best_n_comp``. Also includes ``"best"`` and
        ``"scoring"`` keys.

    Raises:
        ValueError: If ``model_method`` is unsupported or no candidate
            pipeline yields a valid score.
    """
    if model_method != "pls":
        raise ValueError(
            f"Unsupported model_method {model_method!r}; only 'pls' is supported"
        )
    if candidate_pipelines is None:
        candidate_pipelines = list(DEFAULT_CANDIDATE_PIPELINES)
    if candidate_ids is None:
        candidate_ids = [
            f"candidate-{index + 1}" for index in range(len(candidate_pipelines))
        ]
    if len(candidate_ids) != len(candidate_pipelines):
        raise ValueError("candidate_ids length must match candidate_pipelines length")
    if selection_rule not in {"legacy_score", "rmsecv_1pct"}:
        raise ValueError(
            "selection_rule must be 'legacy_score' or 'rmsecv_1pct'; "
            f"got {selection_rule!r}."
        )

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    X_val = np.asarray(X_val, dtype=float)
    y_val = np.asarray(y_val, dtype=float).ravel()

    results: dict = {
        "scoring": (
            "0.7 * cv_r2 + 0.3 * val_r2"
            if selection_rule == "legacy_score"
            else "minimum calibration-only RMSECV with 1% simplicity rule"
        ),
        "selection_rule": selection_rule,
        "candidates": {},
    }

    evaluations: list[dict] = []

    for candidate_index, pipeline in enumerate(candidate_pipelines):
        candidate_id = candidate_ids[candidate_index]
        desc = pipeline.description(locale="zh") or "raw"
        result_key = desc
        if result_key in results["candidates"]:
            result_key = f"{desc} [{candidate_id}]"
        # Fit pipeline on the FULL training set; transform train & val with
        # training statistics so stateful steps cannot leak val info.
        try:
            p_full = pipeline.unfitted_copy().fit(X_train, wv)
            X_tr_pp = p_full.transform(X_train, wv)
            X_val_pp = p_full.transform(X_val, wv)
        except Exception as exc:  # noqa: BLE001 - one candidate is isolated
            evidence = {
                "candidate_id": candidate_id,
                "score": float("-inf"),
                "cv_r2": float("-inf"),
                "cv_rmse": float("inf"),
                "val_r2": float("-inf"),
                "best_n_comp": None,
                "error": str(exc),
            }
            results["candidates"][result_key] = evidence
            evaluations.append(evidence)
            continue

        # Inner CV on RAW TRAIN ONLY (each fold refits the pipeline).
        best_nc, cv_r2, cv_rmse = _pls_cv_best_components(
            pipeline,
            X_train,
            y_train,
            inner_folds,
            max_components,
            random_state,
            wv,
            primary_metric=("rmse" if selection_rule == "rmsecv_1pct" else "r2"),
        )

        # Fit final PLS on full preprocessed train, predict val.
        try:
            m = PLSRegression(n_components=best_nc, scale=False)
            m.fit(X_tr_pp, y_train)
            val_pred = m.predict(X_val_pp).ravel()
            val_r2 = float(r2_score(y_val, val_pred))
        except Exception as exc:  # noqa: BLE001 - one candidate is isolated
            evidence = {
                "candidate_id": candidate_id,
                "score": float("-inf"),
                "cv_r2": float(cv_r2),
                "cv_rmse": float(cv_rmse),
                "val_r2": float("-inf"),
                "best_n_comp": best_nc,
                "error": str(exc),
            }
            results["candidates"][result_key] = evidence
            evaluations.append(evidence)
            continue

        score = 0.7 * float(cv_r2) + 0.3 * float(val_r2)

        evidence = {
            "candidate_id": candidate_id,
            "score": float(score),
            "cv_r2": float(cv_r2),
            "cv_rmse": float(cv_rmse),
            "val_r2": float(val_r2),
            "best_n_comp": int(best_nc),
        }
        results["candidates"][result_key] = evidence
        evaluations.append(evidence)

    winner_index, decision = _select_preprocessing_winner(
        candidate_pipelines,
        evaluations,
        selection_rule=selection_rule,
    )
    best_pipeline = candidate_pipelines[winner_index]
    winner_evidence = evaluations[winner_index]
    results["best"] = best_pipeline.description(locale="zh") or "raw"
    results["best_candidate_id"] = winner_evidence["candidate_id"]
    results["best_score"] = float(winner_evidence["score"])
    results["selection_decision"] = decision
    return best_pipeline, results


def auto_select_components(
    cv_results: dict,
    method: str = "min_rmsECV",
) -> int:
    """Pick an optimal component count from a CV-results dict.

    Args:
        cv_results: Dict produced by :func:`nir_core.model.pls.train_pls`
            containing keys ``"n_components"`` (list[int]) and
            ``"mean_rmse_cv"`` (list[float]).
        method: Selection rule: ``"min_rmsECV"`` (smallest mean RMSECV),
            ``"haaland_thomas"`` (conservative F-test, within 5% of min),
            or ``"first_minimum"`` (first local minimum).

    Returns:
        The selected component count (int).

    Raises:
        ValueError: If required keys are missing, the method is unknown,
            or no finite RMSECV values are present.
    """
    if "n_components" not in cv_results or "mean_rmse_cv" not in cv_results:
        raise ValueError(
            "cv_results must contain 'n_components' and 'mean_rmse_cv' keys"
        )
    n_comp_list = list(cv_results["n_components"])
    mean_rmse = list(cv_results["mean_rmse_cv"])

    if len(n_comp_list) == 0:
        raise ValueError("cv_results['n_components'] is empty")

    rmse_arr = np.asarray(
        [v if np.isfinite(v) else np.inf for v in mean_rmse], dtype=float
    )

    if not np.isfinite(rmse_arr).any():
        raise ValueError("No finite mean_rmse_cv values to select from")

    if method == "min_rmsECV":
        best_idx = int(np.argmin(rmse_arr))
        return int(n_comp_list[best_idx])

    if method == "first_minimum":
        if len(rmse_arr) == 1:
            return int(n_comp_list[0])
        for i in range(1, len(rmse_arr)):
            if rmse_arr[i] > rmse_arr[i - 1]:
                return int(n_comp_list[i - 1])
        return int(n_comp_list[int(np.argmin(rmse_arr))])

    if method == "haaland_thomas":
        min_idx = int(np.argmin(rmse_arr))
        rmse_min = float(rmse_arr[min_idx])
        if rmse_min <= 0:
            return int(n_comp_list[min_idx])
        threshold = 1.05  # within 5% of the minimum (F-ratio ~1.1025)
        for i in range(len(rmse_arr)):
            if not np.isfinite(rmse_arr[i]):
                continue
            ratio = rmse_arr[i] / rmse_min
            if ratio <= threshold:
                return int(n_comp_list[i])
        return int(n_comp_list[min_idx])

    raise ValueError(
        f"Unknown method {method!r}; expected one of "
        f"['min_rmsECV', 'haaland_thomas', 'first_minimum']"
    )
