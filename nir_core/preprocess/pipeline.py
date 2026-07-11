"""Preprocessing pipeline orchestration.

Defines:

- :data:`PRESTEP_METHODS` -- name -> callable mapping used to dispatch
  individual preprocessing steps.
- :class:`PreprocessingPipeline` -- applies an ordered list of
  :class:`nir_core.models.PreprocessingStep` to a spectral matrix. Supports a
  ``fit`` / ``transform`` protocol so that stateful steps (``mean_center``,
  ``autoscale``, ``msc``) compute their statistics on training data only and
  reuse them for validation / test data -- preventing preprocessing leakage in
  nested cross-validation.
- :data:`DEFAULT_CANDIDATE_PIPELINES` -- three representative pipelines
  used by the reflection loop as search starting points.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from nir_core.models import PreprocessingStep

from nir_core.preprocess.scatter import msc, snv
from nir_core.preprocess.smoothing import sg_derivative, sg_smooth
from nir_core.preprocess.baseline import airpls, asls, detrend
from nir_core.preprocess.scaling import autoscale, mean_center, normalize


PRESTEP_METHODS: dict[str, Callable[..., np.ndarray]] = {
    "snv": snv,
    "msc": msc,
    "sg_smooth": sg_smooth,
    "derivative1": lambda X, **kw: sg_derivative(X, deriv=1, **kw),
    "derivative2": lambda X, **kw: sg_derivative(X, deriv=2, **kw),
    "airpls": airpls,
    "asls": asls,
    "detrend": detrend,
    "mean_center": mean_center,
    "autoscale": autoscale,
    "normalize": normalize,
}


# Methods that derive cross-sample statistics and therefore must be "fit" on
# training data before being applied to validation / test data. Without a fit
# phase these methods would compute statistics on whatever data they receive,
# which leaks validation information into the training representation during
# nested cross-validation.
_STATEFUL_METHODS: set[str] = {"mean_center", "autoscale", "msc"}


# Human-readable Chinese labels for the description() method.
_ZH_LABELS: dict[str, str] = {
    "snv": "SNV",
    "msc": "MSC",
    "sg_smooth": "SG平滑",
    "derivative1": "一阶导数",
    "derivative2": "二阶导数",
    "airpls": "airPLS基线校正",
    "asls": "asLS基线校正",
    "detrend": "去趋势",
    "mean_center": "均值中心化",
    "autoscale": "自动缩放",
    "normalize": "归一化",
}


# Parameter keys shown in the description, in display order.
_PARAM_DISPLAY_KEYS: dict[str, list[str]] = {
    "sg_smooth": ["window", "order"],
    "derivative1": ["window", "order"],
    "derivative2": ["window", "order"],
    "airpls": ["lambda_", "porder", "max_iters"],
    "asls": ["lambda_", "p"],
    "normalize": ["norm"],
}


class PreprocessingPipeline:
    """Ordered sequence of preprocessing steps applied to a spectral matrix.

    The pipeline supports two usage modes:

    1. **Stateless / backward-compatible** -- call :meth:`apply` directly.
       Stateful steps (``mean_center``, ``autoscale``, ``msc``) compute their
       statistics from the input array itself. This is convenient for quick
       one-off preprocessing but is *unsafe* in nested CV (it leaks
       validation-fold statistics into the training representation).

    2. **Fit / transform** -- call :meth:`fit` on training data, then
       :meth:`transform` on validation / test data. Stateful steps reuse the
       training statistics, eliminating leakage. This is the mode used by
       :func:`nir_core.model.evaluation.nested_cv_preprocessing`.

    Attributes:
        steps: Ordered list of preprocessing steps.
    """

    def __init__(self, steps: list[PreprocessingStep]) -> None:
        """Construct a pipeline from an ordered list of steps.

        Args:
            steps: Ordered preprocessing steps. Each step's ``method`` must
                be a key in :data:`PRESTEP_METHODS`.
        """
        self.steps: list[PreprocessingStep] = list(steps)
        # Per-step-index fit state for stateful methods. Empty until fit().
        self._fit_state: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Fit / transform protocol (leakage-safe)
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, wv: np.ndarray | None = None) -> "PreprocessingPipeline":
        """Fit stateful steps on training data ``X``.

        Stateless steps are applied in sequence so that stateful steps later
        in the pipeline see the same intermediate representation they would
        see during :meth:`transform`. The fitted state is stored internally
        and reused by :meth:`transform`.

        Args:
            X: Training spectra ``(n_samples, n_wavelengths)``.
            wv: Optional wavelength axis forwarded to ``detrend``.

        Returns:
            ``self`` (fitted pipeline).
        """
        self._fit_state = {}
        out = np.asarray(X, dtype=float)
        for i, step in enumerate(self.steps):
            self._check_method(step.method)
            if step.method in _STATEFUL_METHODS:
                state = self._compute_fit_state(step.method, out)
                self._fit_state[i] = state
                out = self._apply_stateful(step, out, state)
            else:
                out = self._apply_stateless(step, out, wv)
        return self

    def transform(self, X: np.ndarray, wv: np.ndarray | None = None) -> np.ndarray:
        """Apply fitted steps to ``X`` using stored training statistics.

        If the pipeline contains stateful steps but has not been fitted, this
        falls back to :meth:`apply` (computing statistics from ``X`` itself)
        with a ``RuntimeWarning``. Callers doing nested CV should always fit
        first.

        Args:
            X: Spectra to transform ``(n_samples, n_wavelengths)``.
            wv: Optional wavelength axis forwarded to ``detrend``.

        Returns:
            Transformed spectra with the same number of columns as ``X``.
        """
        has_stateful = any(s.method in _STATEFUL_METHODS for s in self.steps)
        if has_stateful and not self._fit_state:
            import warnings

            warnings.warn(
                "PreprocessingPipeline.transform() called without fit() on a pipeline containing stateful steps; falling back to apply() which uses X's own statistics. Call fit() on training data first to prevent leakage.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.apply(X, wv)

        out = np.asarray(X, dtype=float)
        for i, step in enumerate(self.steps):
            if i in self._fit_state:
                out = self._apply_stateful(step, out, self._fit_state[i])
            else:
                out = self._apply_stateless(step, out, wv)
        return out

    # ------------------------------------------------------------------
    # Stateless apply (backward-compatible)
    # ------------------------------------------------------------------
    def apply(self, X: np.ndarray, wv: np.ndarray | None = None) -> np.ndarray:
        """Apply all steps in order to ``X`` using ``X``'s own statistics.

        This is the original stateless behaviour. For stateful steps the
        statistics are derived from ``X`` itself, which is fine for one-off
        preprocessing but **leaks information in nested CV**. Use
        :meth:`fit` + :meth:`transform` instead for model selection.

        Args:
            X: Input spectra ``(n_samples, n_wavelengths)``.
            wv: Optional wavelength axis forwarded to ``detrend``.

        Returns:
            Processed spectra with the same number of columns as ``X``.

        Raises:
            KeyError: If a step's ``method`` is not in
                :data:`PRESTEP_METHODS`.
        """
        out = np.asarray(X, dtype=float)
        for step in self.steps:
            self._check_method(step.method)
            out = self._apply_stateless(step, out, wv)
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _check_method(method: str) -> None:
        if method not in PRESTEP_METHODS:
            raise KeyError(f"Unknown preprocessing method: {method!r}. Available: {sorted(PRESTEP_METHODS.keys())}")

    @staticmethod
    def _apply_stateless(
        step: PreprocessingStep,
        X: np.ndarray,
        wv: np.ndarray | None,
    ) -> np.ndarray:
        params = dict(step.params)
        if step.method == "detrend" and "wv" not in params and wv is not None:
            params["wv"] = wv
        return PRESTEP_METHODS[step.method](X, **params)

    @staticmethod
    def _compute_fit_state(method: str, X: np.ndarray) -> dict:
        """Compute the fit state for a stateful method on ``X``."""
        if method == "mean_center":
            return {"mean": X.mean(axis=0)}
        if method == "autoscale":
            return {
                "mean": X.mean(axis=0),
                "std": X.std(axis=0, ddof=0),
            }
        if method == "msc":
            # MSC reference spectrum = column-wise mean of training data.
            return {"reference": X.mean(axis=0)}
        return {}

    @staticmethod
    def _apply_stateful(
        step: PreprocessingStep,
        X: np.ndarray,
        state: dict,
    ) -> np.ndarray:
        """Apply a stateful method using precomputed ``state``."""
        method = step.method
        if method == "mean_center":
            return X - state["mean"]
        if method == "autoscale":
            std = state["std"]
            safe_std = np.where(std == 0.0, 1.0, std)
            out = (X - state["mean"]) / safe_std
            return np.where(std == 0.0, 0.0, out)
        if method == "msc":
            return msc(X, reference=state["reference"])
        # Should not reach here for non-stateful methods.
        return PRESTEP_METHODS[method](X, **step.params)

    def description(self, locale: str = "zh") -> str:
        """Return a human-readable description of the pipeline.

        Args:
            locale: Language code. Currently only ``"zh"`` (Chinese) is
                supported; any other value falls back to the Chinese labels.

        Returns:
            A string like ``"SNV → SG平滑(window=11,order=2) → 均值中心化"``.
        """
        parts: list[str] = []
        for step in self.steps:
            label = _ZH_LABELS.get(step.method, step.method)
            param_keys = _PARAM_DISPLAY_KEYS.get(step.method, [])
            param_strs = []
            for k in param_keys:
                if k in step.params:
                    param_strs.append(f"{k}={step.params[k]}")
            if param_strs:
                label = f"{label}({','.join(param_strs)})"
            parts.append(label)
        return " → ".join(parts)

    def fitted(self) -> bool:
        """Return True if :meth:`fit` has been called with stateful state."""
        return bool(self._fit_state)


DEFAULT_CANDIDATE_PIPELINES: list[PreprocessingPipeline] = [
    PreprocessingPipeline(
        steps=[
            PreprocessingStep(method="snv", params={}),
            PreprocessingStep(method="sg_smooth", params={"window": 11, "order": 2}),
            PreprocessingStep(method="mean_center", params={}),
        ]
    ),
    PreprocessingPipeline(
        steps=[
            PreprocessingStep(method="msc", params={}),
            PreprocessingStep(method="derivative1", params={"window": 11, "order": 2}),
            PreprocessingStep(method="autoscale", params={}),
        ]
    ),
    PreprocessingPipeline(
        steps=[
            PreprocessingStep(method="airpls", params={"lambda_": 1e7}),
            PreprocessingStep(method="sg_smooth", params={"window": 11, "order": 2}),
            PreprocessingStep(method="snv", params={}),
            PreprocessingStep(method="mean_center", params={}),
        ]
    ),
]


# ---------------------------------------------------------------------------
# Pipeline validation (V3.6: flexible guardrail for LLM-generated pipelines)
# ---------------------------------------------------------------------------

# Pipeline stage ordering: lower stage numbers run first.
# Stage 0: baseline correction (must come first).
# Stage 1: scatter correction + smoothing/derivative (interleavable).
# Stage 2: scaling (must come last).
_STAGE_ORDER: dict[str, int] = {
    "airpls": 0,
    "asls": 0,
    "detrend": 0,
    "snv": 1,
    "msc": 1,
    "sg_smooth": 1,
    "derivative1": 1,
    "derivative2": 1,
    "mean_center": 2,
    "autoscale": 2,
    "normalize": 2,
}

# Methods that cannot coexist in the same pipeline.
_EXCLUSIVE_GROUPS: list[set[str]] = [
    {"snv", "msc"},  # both are scatter correction
    {"derivative1", "derivative2"},  # one derivative at a time
    {"airpls", "asls"},  # both are baseline correction
    {"mean_center", "autoscale", "normalize"},  # one scaling method
]

# Parameter range constraints: {method: {param: (min, max, extra_check)}}
_PARAM_CONSTRAINTS: dict[str, dict[str, tuple]] = {
    "sg_smooth": {"window": (5, 15), "order": (1, 4)},
    "derivative1": {"window": (5, 15), "order": (1, 4)},
    "derivative2": {"window": (5, 15), "order": (2, 4)},
    "airpls": {"lambda_": (1e3, 1e8)},
    "asls": {"lambda_": (1e3, 1e8), "p": (0.001, 0.999)},
}


def validate_pipeline(steps: list[PreprocessingStep]) -> tuple[bool, str]:
    """Validate a preprocessing pipeline for chemometric correctness.

    Checks three categories of constraints:

    1. **Whitelist**: every method must be in :data:`PRESTEP_METHODS`.
    2. **Order**: methods should follow the standard stage sequence
       (baseline → scatter → smoothing/derivative → scaling). Out-of-order
       warnings are returned as failures to prevent nonsensical pipelines.
    3. **Exclusivity**: mutually exclusive methods (e.g. ``snv`` and ``msc``)
       cannot appear in the same pipeline.
    4. **Parameter ranges**: hyper-parameters must fall within safe bounds
       (e.g. SG window 5–15 and odd, airPLS lambda 1e3–1e8).

    Args:
        steps: Ordered list of preprocessing steps to validate.

    Returns:
        ``(is_valid, reason)`` — ``reason`` is empty when valid, otherwise a
        Chinese description of the first violation found.
    """
    if not steps:
        return True, ""

    methods = [s.method for s in steps]

    # 1. Whitelist check.
    for m in methods:
        if m not in PRESTEP_METHODS:
            return False, f"未知预处理方法: {m!r}，可用方法: {sorted(PRESTEP_METHODS.keys())}"

    # 2. Order check.
    prev_stage = -1
    for i, m in enumerate(methods):
        stage = _STAGE_ORDER.get(m, -1)
        if stage < prev_stage:
            return False, (f"方法顺序不当: {m!r}（阶段{stage}）出现在更晚阶段的方法之后。建议顺序: 基线校正 → 散射校正 → 平滑/求导 → 缩放。")
        prev_stage = max(prev_stage, stage)

    # 3. Exclusivity check.
    method_set = set(methods)
    for group in _EXCLUSIVE_GROUPS:
        clash = method_set & group
        if len(clash) > 1:
            return False, f"互斥方法不能同时存在: {sorted(clash)}"

    # 4. Parameter range check.
    for step in steps:
        constraints = _PARAM_CONSTRAINTS.get(step.method)
        if not constraints:
            continue
        for param, (lo, hi) in constraints.items():
            val = step.params.get(param)
            if val is None:
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                return False, f"参数 {param}={val!r} 不是有效数值 (方法 {step.method!r})"
            if val < lo or val > hi:
                return False, f"参数 {param}={val} 超出范围 [{lo}, {hi}] (方法 {step.method!r})"
            # SG window must be odd.
            if param == "window" and int(val) != val:
                return False, f"SG 窗口必须是整数 (方法 {step.method!r})"
            if param == "window" and int(val) % 2 == 0:
                return False, f"SG 窗口必须是奇数 (当前值 {int(val)})"

    return True, ""


__all__ = [
    "PRESTEP_METHODS",
    "PreprocessingPipeline",
    "DEFAULT_CANDIDATE_PIPELINES",
    "validate_pipeline",
]
