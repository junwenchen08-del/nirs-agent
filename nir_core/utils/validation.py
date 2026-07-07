"""Validation utilities: outlier detection, leakage checks, retry logic.

Functions are pure (no side effects) except ``get_next_pipeline`` which
reads the global ``NirConfig`` to enumerate candidate preprocessing
pipelines. Outlier detectors return index arrays so callers can decide
how to handle flagged samples.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from nir_core.config import get_nir_config
from nir_core.models import PreprocessingStep

try:  # sklearn is the preferred backend but may be binary-incompatible
    from sklearn.decomposition import PCA as _SklearnPCA

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - environment-dependent
    _SklearnPCA = None
    _HAS_SKLEARN = False


def _pca_fit_transform(X: np.ndarray, n_components: int):
    """Fit a PCA and return (scores, explained_variance).

    Uses sklearn when available; otherwise falls back to a NumPy SVD-based
    PCA. ``explained_variance`` follows the sklearn convention
    (variance of each score column = eigenvalue / (n_samples - 1)).
    """
    if _HAS_SKLEARN:
        pca = _SklearnPCA(n_components=n_components)
        scores = pca.fit_transform(X)
        return scores, pca.explained_variance_
    # ---- NumPy fallback (compact SVD on centered data) ----
    Xc = X - np.mean(X, axis=0, keepdims=True)
    # economy SVD: Xc = U S V^T
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    n_samples = Xc.shape[0]
    explained_variance = (s ** 2) / max(1, n_samples - 1)
    k = min(n_components, s.shape[0])
    scores = U[:, :k] * s[:k]
    return scores, explained_variance[:k]


def detect_outliers_pca_t2(
    X: np.ndarray, n_components: int = 5, alpha: float = 0.95
) -> np.ndarray:
    """Detect outliers via Hotelling's T^2 statistic on PCA scores.

    Fits a PCA with ``n_components`` on ``X``, computes
    ``T^2 = sum_i (t_i^2 / lambda_i)`` for each sample, and flags samples
    whose T^2 exceeds the F-distribution critical value at level ``alpha``.

    Args:
        X: Spectral matrix, shape (n_samples, n_wavelengths).
        n_components: Number of PCA components to retain.
        alpha: Confidence level for the F critical value (e.g. 0.95).

    Returns:
        1-D int array of outlier sample indices (possibly empty).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array.")
    n_samples = X.shape[0]
    if n_samples < 2:
        return np.array([], dtype=int)

    # Clamp components to feasible range.
    k = max(1, min(n_components, X.shape[1], n_samples - 1))
    scores, lambdas = _pca_fit_transform(X, k)
    # Guard against zero/negative eigenvalues from numerical noise.
    lambdas = np.where(lambdas > 1e-12, lambdas, 1e-12)

    t2 = np.sum((scores ** 2) / lambdas, axis=1)

    # Critical value: F(alpha, p, n-p-1) * p * (n-1) / (n-p).
    p = k
    dfn = p
    dfd = max(1, n_samples - p - 1)
    f_crit = float(stats.f.ppf(alpha, dfn, dfd))
    threshold = f_crit * p * (n_samples - 1) / max(1, (n_samples - p))

    flagged = np.where(t2 > threshold)[0]
    return flagged.astype(int)


def detect_outliers_mahalanobis(
    X: np.ndarray, threshold: float = 3.0
) -> np.ndarray:
    """Detect outliers via Mahalanobis distance from the column mean.

    Uses the pseudo-inverse of the covariance matrix (``np.linalg.pinv``)
    to remain stable for high-dimensional / singular covariance cases.
    Distances are compared to ``threshold`` (default 3.0 corresponds to a
    loose 3-sigma rule on the chi distribution).

    Args:
        X: Spectral matrix, shape (n_samples, n_wavelengths).
        threshold: Distance threshold; samples above are flagged.

    Returns:
        1-D int array of outlier sample indices.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array.")
    n_samples = X.shape[0]
    if n_samples < 2:
        return np.array([], dtype=int)

    mean_vec = np.mean(X, axis=0)
    centered = X - mean_vec
    cov = np.cov(centered, rowvar=False)
    # cov may be scalar when n_wavelengths==1; ensure 2-D.
    cov = np.atleast_2d(cov)
    cov_inv = np.linalg.pinv(cov)

    # Mahalanobis distance per sample.
    diff = centered
    # d_i = sqrt(diff_i @ cov_inv @ diff_i)
    mahal = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, cov_inv, diff), 0.0))

    flagged = np.where(mahal > threshold)[0]
    return flagged.astype(int)


def check_train_test_split_leakage(
    X_train: np.ndarray, X_test: np.ndarray
) -> bool:
    """Check whether any test row exactly duplicates a training row.

    Uses a row-hash set for efficiency; exact equality is required (a
    single shared row constitutes leakage).

    Args:
        X_train: Training matrix, shape (n_train, n_features).
        X_test: Test matrix, shape (n_test, n_features).

    Returns:
        True if at least one identical row appears in both sets.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Both inputs must be 2-D arrays.")
    if X_train.shape[1] != X_test.shape[1]:
        return False

    def _row_hash(row: np.ndarray) -> bytes:
        return row.tobytes()

    train_hashes = {_row_hash(row) for row in X_train}
    for row in X_test:
        if _row_hash(row) in train_hashes:
            return True
    return False


def should_retry(
    quality_result: dict,
    attempt: int,
    max_retries: int = 3,
    history: list[dict] | None = None,
) -> bool:
    """Decide whether the reflection loop should retry with a new pipeline.

    Rules (evaluated in order):
        1. If the quality gate passed (``quality_result["passed"]`` is True),
           do not retry.
        2. If ``attempt >= max_retries``, do not retry (budget exhausted).
        3. If the last two history entries have R² values differing by less
           than 0.02 (plateau), do not retry.
        4. Otherwise retry.

    Args:
        quality_result: Output of ``evaluate_quality`` (must contain
            ``"passed"``).
        attempt: Current 1-based attempt number.
        max_retries: Maximum allowed retries.
        history: Optional list of prior attempt records, each containing a
            ``"metrics"`` dict with an R² value under ``R2_val`` or ``R2``.

    Returns:
        True if a retry should be attempted.
    """
    if quality_result.get("passed", False):
        return False
    if attempt >= max_retries:
        return False
    if history and len(history) >= 2:
        last_two = history[-2:]

        def _r2(rec: dict) -> float | None:
            m = rec.get("metrics", {}) if isinstance(rec, dict) else {}
            for key in ("R2_val", "R2"):
                val = m.get(key)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        return None
            return None

        r2_a = _r2(last_two[-2])
        r2_b = _r2(last_two[-1])
        if r2_a is not None and r2_b is not None:
            if abs(r2_b - r2_a) < 0.02:
                return False
    return True


def _pipeline_signature(steps: list[PreprocessingStep]) -> tuple[str, ...]:
    """Build a hashable signature tuple from a list of PreprocessingStep."""
    return tuple(s.method for s in steps)


def _history_pipeline_signatures(history: list[dict] | None) -> set[tuple[str, ...]]:
    """Extract already-tried pipeline signatures from history records.

    Each history entry may carry a ``"pipeline"`` key holding a list of
    dicts with a ``"method"`` field (the serialized PreprocessingStep form).
    """
    sigs: set[tuple[str, ...]] = set()
    if not history:
        return sigs
    for rec in history:
        if not isinstance(rec, dict):
            continue
        pipeline = rec.get("pipeline")
        if not isinstance(pipeline, list):
            continue
        methods: list[str] = []
        for step in pipeline:
            if isinstance(step, PreprocessingStep):
                methods.append(step.method)
            elif isinstance(step, dict) and "method" in step:
                methods.append(str(step["method"]))
        if methods:
            sigs.add(tuple(methods))
    return sigs


def get_next_pipeline(
    attempt: int,
    history: list[dict] | None = None,
) -> list[PreprocessingStep] | None:
    """Return the next untried candidate preprocessing pipeline.

    Reads ``get_nir_config().candidate_pipelines`` (a list of method-name
    lists) and returns the first pipeline whose method sequence has not
    yet appeared in ``history``. Each candidate is converted to a list of
    ``PreprocessingStep`` objects with empty params.

    Args:
        attempt: Current attempt number (unused for selection logic but
            kept for API symmetry with ``should_retry``).
        history: Prior attempt records (see ``_history_pipeline_signatures``).

    Returns:
        A list of ``PreprocessingStep``, or ``None`` if every candidate
        pipeline has already been tried.
    """
    cfg = get_nir_config()
    candidates = cfg.candidate_pipelines or []
    tried = _history_pipeline_signatures(history)

    for candidate in candidates:
        sig = tuple(candidate)
        if sig in tried:
            continue
        return [PreprocessingStep(method=str(m)) for m in candidate]
    return None


def suggest_lv_adjustment(
    metrics: dict,
    domain: str = "default",
    n_samples: int | None = None,
) -> dict:
    """Suggest a latent-variable (n_components) adjustment based on diagnostics.

    Combines overfitting detection, bias analysis, and underfitting checks
    to produce a concrete recommendation for the next modelling attempt.

    Decision logic:
        1. **Overfitting** (RMSEP > 2×RMSECV): suggest reducing n_components
           by 1–2 (floor at 1). Also recommend retrying with a stronger
           preprocessing pipeline.
        2. **Bias issue** (|bias| > 0.1×|mean(y)|): recommend bias correction
           (post-hoc additive correction on predictions) rather than LV change.
        3. **Underfitting** (R² < min_r2 − 0.20): suggest increasing
           n_components by 1 (cap at max(20, n_samples//10)) **and** retrying
           preprocessing.
        4. **Healthy** (passed + no overfitting): no change needed.

    Args:
        metrics: Full metrics dict (must contain ``n_components``, and
            preferably ``RMSEP``, ``RMSECV``, ``test.bias``, ``R2_val``).
        domain: Application domain for threshold lookup.
        n_samples: Sample count (for small-sample relaxation).

    Returns:
        Dict with keys:
        - ``current_n``: current n_components (or None).
        - ``suggested_n``: suggested n_components (or None if no change).
        - ``change``: one of ``{"keep", "reduce", "increase", "none"}``.
        - ``delta``: signed delta (e.g. -2, +1, 0).
        - ``issues``: list of detected issue strings.
        - ``recommendation``: Chinese recommendation text.
        - ``retry_preprocessing``: bool — whether a new pipeline is advised.
    """
    from nir_core.utils.metrics import evaluate_quality  # avoid cycle

    quality = evaluate_quality(metrics, domain=domain, n_samples=n_samples)
    current_n = metrics.get("n_components")
    if current_n is not None:
        try:
            current_n = int(current_n)
        except (TypeError, ValueError):
            current_n = None

    issues: list[str] = []
    retry_pp = False
    suggested_n = current_n
    change = "keep"
    delta = 0

    # --- 1. Overfitting detection ---
    rmsep = metrics.get("RMSEP")
    rmsecv = metrics.get("RMSECV")
    overfitting = False
    if rmsep is not None and rmsecv is not None:
        try:
            rmsep_f = float(rmsep)
            rmsecv_f = float(rmsecv)
            if rmsecv_f > 0 and rmsep_f > 2.0 * rmsecv_f:
                overfitting = True
                issues.append("overfitting")
        except (TypeError, ValueError):
            pass

    if overfitting and current_n is not None and current_n > 1:
        # Reduce by up to 2, floor at 1.
        delta = -min(2, current_n - 1)
        suggested_n = current_n + delta
        change = "reduce"
        retry_pp = True

    # --- 2. Bias issue detection ---
    test_block = metrics.get("test")
    test_bias = 0.0
    if isinstance(test_block, dict):
        try:
            test_bias = float(test_block.get("bias", 0.0))
        except (TypeError, ValueError):
            test_bias = 0.0
    y_vals = metrics.get("y")
    ref_central = 1.0
    if y_vals is not None:
        try:
            ref_central = float(np.mean(np.asarray(y_vals, dtype=float)))
        except (TypeError, ValueError):
            ref_central = 1.0
    if ref_central == 0.0:
        ref_central = 1.0
    bias_issue = (test_bias != 0.0) and (abs(test_bias) > 0.1 * abs(ref_central))
    if bias_issue:
        issues.append("bias_issue")

    # --- 3. Underfitting detection ---
    r2 = metrics.get("R2_val")
    if r2 is None:
        r2 = metrics.get("R2")
    min_r2 = quality["thresholds_used"]["min_r2"]
    underfitting = False
    if r2 is not None:
        try:
            r2_f = float(r2)
            if r2_f < min_r2 - 0.20:
                underfitting = True
                issues.append("underfitting")
        except (TypeError, ValueError):
            pass

    if underfitting and current_n is not None and change == "keep":
        # Increase by 1, cap at max(20, n_samples//10).
        cap = 20
        if n_samples is not None:
            cap = max(20, n_samples // 10)
        if current_n < cap:
            delta = 1
            suggested_n = current_n + delta
            change = "increase"
        retry_pp = True

    # --- 4. Healthy ---
    if not issues and quality["passed"]:
        change = "none"
        delta = 0
        suggested_n = current_n

    # Recommendation text.
    rec_parts: list[str] = []
    if change == "none":
        rec_parts.append("模型健康，无需调整潜变量数。")
    elif change == "reduce":
        rec_parts.append(
            f"检测到过拟合(RMSEP={rmsep:.4f} > 2×RMSECV={rmsecv:.4f})，"
            f"建议将成分数从 {current_n} 减少到 {suggested_n}。"
        )
    elif change == "increase":
        rec_parts.append(
            f"检测到欠拟合(R²={r2_f:.4f} < {min_r2 - 0.20:.2f})，"
            f"建议将成分数从 {current_n} 增加到 {suggested_n}。"
        )
    if bias_issue:
        rec_parts.append(
            f"检测到显著系统偏差(bias={test_bias:.4f})，"
            f"建议在预测后做偏差校正: y_corrected = y_pred - {test_bias:.4f}。"
        )
    if retry_pp:
        rec_parts.append("同时建议更换预处理流水线后重试。")

    return {
        "current_n": current_n,
        "suggested_n": suggested_n,
        "change": change,
        "delta": delta,
        "issues": issues,
        "recommendation": " ".join(rec_parts) if rec_parts else "无需调整。",
        "retry_preprocessing": retry_pp,
        "quality": quality,
    }
