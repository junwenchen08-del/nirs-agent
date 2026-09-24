"""Preprocessing pipeline orchestration.

Defines:

- :data:`PRESTEP_METHODS` -- name -> callable mapping used to dispatch
  individual preprocessing steps.
- :class:`PreprocessingPipeline` -- applies an ordered list of
  :class:`nir_core.models.PreprocessingStep` to a spectral matrix. Supports a
  ``fit`` / ``transform`` protocol so that stateful steps (``mean_center``,
  ``autoscale``, ``msc``, ``emsc``) compute their statistics on training data
  only and reuse them for validation / test data -- preventing preprocessing leakage in
  nested cross-validation.
- :data:`DEFAULT_CANDIDATE_PIPELINES` -- three representative pipelines
  used by the reflection loop as search starting points.
"""

from __future__ import annotations

import os

import numpy as np

from nir_core.models import PreprocessingStep
from nir_core.preprocess.registry import (
    METHOD_REGISTRY,
    build_exclusive_groups,
    get_provider_implementation,
    resolve_preferred_provider,
    validate_method_params,
)

PRESTEP_METHODS = {
    method_id: spec.implementation for method_id, spec in METHOD_REGISTRY.items()
}


# Methods that derive cross-sample statistics and therefore must be "fit" on
# training data before being applied to validation / test data. Without a fit
# phase these methods would compute statistics on whatever data they receive,
# which leaks validation information into the training representation during
# nested cross-validation.
_STATEFUL_METHODS: set[str] = {
    method_id for method_id, spec in METHOD_REGISTRY.items() if spec.stateful
}


# Human-readable Chinese labels for the description() method.
_ZH_LABELS: dict[str, str] = {
    method_id: spec.display_name_zh for method_id, spec in METHOD_REGISTRY.items()
}


# Parameter keys shown in the description, in display order.
_PARAM_DISPLAY_KEYS: dict[str, list[str]] = {
    method_id: [
        name for name, parameter in spec.parameter_schema.items() if parameter.display
    ]
    for method_id, spec in METHOD_REGISTRY.items()
    if spec.parameter_schema
}


class PreprocessingPipeline:
    """Ordered sequence of preprocessing steps applied to a spectral matrix.

    The pipeline supports two usage modes:

    1. **Stateless / backward-compatible** -- call :meth:`apply` directly.
       Stateful steps (``mean_center``, ``autoscale``, ``msc``, ``emsc``) compute their
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

    def __init__(
        self,
        steps: list[PreprocessingStep],
        provider_policy: str | None = None,
    ) -> None:
        """Construct a pipeline from an ordered list of steps.

        Args:
            steps: Ordered preprocessing steps. Each step's ``method`` must
                be a key in :data:`PRESTEP_METHODS`.
            provider_policy: ``"catalog_default"`` selects each method's
                verified preferred provider. ``"native"`` pins every step
                to the original project implementation. When omitted, the
                ``NIR_PREPROCESSING_PROVIDER_POLICY`` environment variable is
                honored and otherwise defaults to ``"catalog_default"``.
        """
        self.steps: list[PreprocessingStep] = list(steps)
        selected_policy = (
            (
                provider_policy
                or os.getenv(
                    "NIR_PREPROCESSING_PROVIDER_POLICY",
                    "catalog_default",
                )
            )
            .strip()
            .lower()
        )
        if selected_policy not in {"catalog_default", "native"}:
            raise ValueError(
                "provider_policy must be 'catalog_default' or 'native'; "
                f"got {selected_policy!r}."
            )
        self._provider_policy = selected_policy
        self._provider_bindings: list[str] = []
        for step in self.steps:
            spec = METHOD_REGISTRY.get(step.method)
            if spec is None:
                # Preserve the historical behavior: an unknown method fails
                # when the pipeline is validated or executed, not while its
                # lightweight container is being constructed.
                self._provider_bindings.append("native")
                continue
            provider = (
                "native"
                if selected_policy == "native"
                else resolve_preferred_provider(step.method, step.params)
            )
            get_provider_implementation(step.method, provider)
            self._provider_bindings.append(provider)
        # Per-step-index fit state for stateful methods. Empty until fit().
        self._fit_state: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Fit / transform protocol (leakage-safe)
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, wv: np.ndarray | None = None) -> PreprocessingPipeline:
        """Fit stateful steps on training data ``X``.

        Stateless steps are applied in sequence so that stateful steps later
        in the pipeline see the same intermediate representation they would
        see during :meth:`transform`. The fitted state is stored internally
        and reused by :meth:`transform`.

        Args:
            X: Training spectra ``(n_samples, n_wavelengths)``.
            wv: Optional wavelength axis forwarded to wavelength-aware
                methods such as ``detrend`` and ``emsc``.

        Returns:
            ``self`` (fitted pipeline).
        """
        self._fit_state = {}
        out = np.asarray(X, dtype=float)
        for i, step in enumerate(self.steps):
            self._check_method(step.method)
            if step.method in _STATEFUL_METHODS:
                state = self._compute_fit_state(i, step, out, wv)
                self._fit_state[i] = state
                out = self._apply_stateful(i, step, out, state, wv)
            else:
                out = self._apply_stateless(i, step, out, wv)
        return self

    def transform(self, X: np.ndarray, wv: np.ndarray | None = None) -> np.ndarray:
        """Apply fitted steps to ``X`` using stored training statistics.

        If the pipeline contains stateful steps but has not been fitted, this
        falls back to :meth:`apply` (computing statistics from ``X`` itself)
        with a ``RuntimeWarning``. Callers doing nested CV should always fit
        first.

        Args:
            X: Spectra to transform ``(n_samples, n_wavelengths)``.
            wv: Optional wavelength axis forwarded to wavelength-aware
                methods such as ``detrend`` and ``emsc``.

        Returns:
            Transformed spectra with the same number of columns as ``X``.
        """
        has_stateful = any(s.method in _STATEFUL_METHODS for s in self.steps)
        fit_state = getattr(self, "_fit_state", {})
        if has_stateful and not fit_state:
            import warnings

            warnings.warn(
                "PreprocessingPipeline.transform() called without fit() on a pipeline containing stateful steps; falling back to apply() which uses X's own statistics. Call fit() on training data first to prevent leakage.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.apply(X, wv)

        out = np.asarray(X, dtype=float)
        for i, step in enumerate(self.steps):
            if i in fit_state:
                out = self._apply_stateful(i, step, out, fit_state[i], wv)
            else:
                out = self._apply_stateless(i, step, out, wv)
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
            wv: Optional wavelength axis forwarded to wavelength-aware
                methods such as ``detrend`` and ``emsc``.

        Returns:
            Processed spectra with the same number of columns as ``X``.

        Raises:
            KeyError: If a step's ``method`` is not in
                :data:`PRESTEP_METHODS`.
        """
        out = np.asarray(X, dtype=float)
        for i, step in enumerate(self.steps):
            self._check_method(step.method)
            out = self._apply_stateless(i, step, out, wv)
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _check_method(method: str) -> None:
        if method not in PRESTEP_METHODS:
            raise KeyError(
                f"Unknown preprocessing method: {method!r}. Available: {sorted(PRESTEP_METHODS.keys())}"
            )

    def _provider_for_step(self, index: int) -> str:
        """Return the persisted provider, pinning legacy artifacts to native."""
        bindings = getattr(self, "_provider_bindings", None)
        if bindings is None or index >= len(bindings):
            return "native"
        return bindings[index]

    def _implementation_for_step(self, index: int, method: str):
        provider = self._provider_for_step(index)
        return get_provider_implementation(method, provider).implementation

    def _apply_stateless(
        self,
        index: int,
        step: PreprocessingStep,
        X: np.ndarray,
        wv: np.ndarray | None,
    ) -> np.ndarray:
        params = dict(step.params)
        if step.method in {"detrend", "emsc"} and "wv" not in params and wv is not None:
            params["wv"] = wv
        if (
            step.method in {"derivative1", "derivative2"}
            and params.get("spacing_mode") == "wavelength"
            and "wv" not in params
            and wv is not None
        ):
            params["wv"] = wv
        implementation = self._implementation_for_step(index, step.method)
        return implementation(X, **params)

    def _compute_fit_state(
        self,
        index: int,
        step: PreprocessingStep,
        X: np.ndarray,
        wv: np.ndarray | None,
    ) -> dict:
        """Compute the fit state for a stateful method on ``X``."""
        method = step.method
        provider = get_provider_implementation(
            method,
            self._provider_for_step(index),
        )
        if provider.fit_state is not None:
            if provider.transform_state is None:
                raise RuntimeError(
                    f"Provider {provider.provider!r} for {method!r} defines "
                    "fit_state without transform_state."
                )
            return {
                "provider_managed": True,
                "provider_state": provider.fit_state(
                    X,
                    params=dict(step.params),
                    wv=wv,
                ),
            }
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
        if method == "emsc":
            return {
                "reference": X.mean(axis=0),
                "wv": None if wv is None else np.asarray(wv, dtype=float).copy(),
            }
        return {}

    def _apply_stateful(
        self,
        index: int,
        step: PreprocessingStep,
        X: np.ndarray,
        state: dict,
        wv: np.ndarray | None,
    ) -> np.ndarray:
        """Apply a stateful method using precomputed ``state``."""
        method = step.method
        if state.get("provider_managed"):
            provider = get_provider_implementation(
                method,
                self._provider_for_step(index),
            )
            if provider.transform_state is None:
                raise RuntimeError(
                    f"Provider {provider.provider!r} for {method!r} cannot "
                    "replay fitted state."
                )
            return provider.transform_state(
                state["provider_state"],
                X,
                params=dict(step.params),
                wv=wv,
            )
        if method == "mean_center":
            return X - state["mean"]
        if method == "autoscale":
            std = state["std"]
            safe_std = np.where(std == 0.0, 1.0, std)
            out = (X - state["mean"]) / safe_std
            return np.where(std == 0.0, 0.0, out)
        if method == "msc":
            implementation = self._implementation_for_step(index, method)
            return implementation(X, reference=state["reference"])
        if method == "emsc":
            fitted_wv = state.get("wv")
            if fitted_wv is not None:
                if wv is None:
                    raise ValueError(
                        "EMSC was fitted with a wavelength axis; transform() "
                        "must receive the same axis."
                    )
                current_wv = np.asarray(wv, dtype=float).ravel()
                if current_wv.shape != fitted_wv.shape or not np.allclose(
                    current_wv, fitted_wv, rtol=0.0, atol=1e-10
                ):
                    raise ValueError(
                        "EMSC transform wavelength axis does not match the fitted axis."
                    )
            params = dict(step.params)
            params.pop("wv", None)
            implementation = self._implementation_for_step(index, method)
            return implementation(
                X,
                reference=state["reference"],
                wv=fitted_wv,
                **params,
            )
        # Should not reach here for non-stateful methods.
        implementation = self._implementation_for_step(index, method)
        return implementation(X, **step.params)

    def provider_manifest(self) -> list[dict[str, object]]:
        """Return the exact implementation binding for every pipeline step.

        Older serialized pipelines do not contain ``_provider_bindings``.
        They are deliberately reported and executed as ``native`` so loading
        a historical model cannot silently change its numerical behavior.
        """
        manifest: list[dict[str, object]] = []
        for index, step in enumerate(self.steps):
            provider = self._provider_for_step(index)
            implementation = get_provider_implementation(step.method, provider)
            preferred = resolve_preferred_provider(step.method, step.params)
            manifest.append(
                {
                    "step_index": index,
                    "method_id": step.method,
                    **implementation.as_dict(is_default=provider == preferred),
                }
            )
        return manifest

    def unfitted_copy(self) -> PreprocessingPipeline:
        """Clone steps and exact provider bindings without fitted statistics."""
        clone = PreprocessingPipeline(self.steps, provider_policy="native")
        bindings = getattr(self, "_provider_bindings", None)
        if bindings is None or len(bindings) != len(self.steps):
            clone._provider_policy = "legacy_native"
            clone._provider_bindings = ["native"] * len(self.steps)
        else:
            clone._provider_policy = getattr(
                self,
                "_provider_policy",
                "legacy_native",
            )
            clone._provider_bindings = list(bindings)
        return clone

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

    @property
    def has_stateful_steps(self) -> bool:
        """Return True if any step derives cross-sample statistics."""
        return any(step.method in _STATEFUL_METHODS for step in self.steps)


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
# Stage -1: isolated-spike correction (must come first when present).
# Stage 0: baseline correction.
# Stage 1: scatter correction + smoothing/derivative (interleavable).
# Stage 2: scaling (must come last).
_STAGE_ORDER: dict[str, int] = {
    method_id: spec.stage_order for method_id, spec in METHOD_REGISTRY.items()
}

# Methods that cannot coexist in the same pipeline.
_EXCLUSIVE_GROUPS: list[set[str]] = build_exclusive_groups()


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
            return (
                False,
                f"未知预处理方法: {m!r}，可用方法: {sorted(PRESTEP_METHODS.keys())}",
            )

    # 2. Order check.
    prev_stage = -1
    for i, m in enumerate(methods):
        stage = _STAGE_ORDER.get(m, -1)
        if stage < prev_stage:
            return False, (
                f"方法顺序不当: {m!r}（阶段{stage}）出现在更晚阶段的方法之后。建议顺序: 去尖峰 → 基线校正 → 散射校正 → 平滑/求导 → 缩放。"
            )
        prev_stage = max(prev_stage, stage)

    # 3. Exclusivity check.
    method_set = set(methods)
    for group in _EXCLUSIVE_GROUPS:
        clash = method_set & group
        if len(clash) > 1:
            return False, f"互斥方法不能同时存在: {sorted(clash)}"

    # 4. Parameter schema check.
    for step in steps:
        is_valid, reason = validate_method_params(step.method, step.params)
        if not is_valid:
            return False, reason

    return True, ""


__all__ = [
    "DEFAULT_CANDIDATE_PIPELINES",
    "PRESTEP_METHODS",
    "PreprocessingPipeline",
    "validate_pipeline",
]
