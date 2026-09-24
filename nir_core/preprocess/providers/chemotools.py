"""Compatibility adapters for the pinned Chemotools 0.4.4 backend.

The public signatures intentionally follow ``nir_core`` rather than exposing
upstream constructor names.  Adapters also retain project safety semantics
for one-dimensional input and degenerate spectra.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
from chemotools.baseline import (
    AirPls,
    ArPls,
    AsLs,
    PolynomialCorrection,
    RubberbandCorrection,
)
from chemotools.derivative import NorrisWilliams, SavitzkyGolay
from chemotools.scale import NormScaler, ParetoScaler
from chemotools.scatter import (
    ExtendedMultiplicativeScatterCorrection,
    MultiplicativeScatterCorrection,
    RobustNormalVariate,
    StandardNormalVariate,
)
from chemotools.smooth import MedianFilter, SavitzkyGolayFilter, WhittakerSmooth

from nir_core.preprocess.scatter import msc as native_msc

PROVIDER_NAME = "chemotools"
PROVIDER_VERSION = "0.4.4"
IMPLEMENTATION_VERSION = "chemotools-adapter-v1"
EXTENDED_IMPLEMENTATION_VERSION = "chemotools-adapter-v2"


def _ensure_2d(X: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(X, dtype=float)
    if array.ndim == 1:
        return array[np.newaxis, :], True
    if array.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got {array.ndim}D.")
    return array, False


def _restore_shape(X: np.ndarray, was_1d: bool) -> np.ndarray:
    result = np.asarray(X, dtype=float)
    return result[0] if was_1d else result


def snv(X: np.ndarray) -> np.ndarray:
    """Apply Chemotools SNV while keeping constant rows finite and zero."""
    array, was_1d = _ensure_2d(X)
    std = array.std(axis=1, ddof=0)
    result = np.zeros_like(array)
    nonconstant = std > 0.0
    if np.any(nonconstant):
        transformer = StandardNormalVariate()
        result[nonconstant] = transformer.fit_transform(array[nonconstant])
    return _restore_shape(result, was_1d)


def rnv(
    X: np.ndarray,
    percentile: float = 25.0,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Apply Chemotools robust normal variate as its own public method."""
    array, was_1d = _ensure_2d(X)
    transformer = RobustNormalVariate(
        percentile=percentile,
        epsilon=epsilon,
        n_jobs=1,
    )
    result = transformer.fit_transform(array)
    if not np.isfinite(result).all():
        raise ValueError("Chemotools RNV produced non-finite output.")
    return _restore_shape(result, was_1d)


def msc(
    X: np.ndarray,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Apply Chemotools MSC with the project's degenerate-input fallback."""
    array, was_1d = _ensure_2d(X)
    resolved_reference = (
        array.mean(axis=0)
        if reference is None
        else np.asarray(reference, dtype=float).ravel()
    )
    if resolved_reference.shape[0] != array.shape[1]:
        raise ValueError(
            f"reference length {resolved_reference.shape[0]} does not match "
            f"n_wavelengths {array.shape[1]}."
        )

    # Chemotools correctly handles ordinary MSC but its normal-equation fit is
    # singular for a constant reference and divides by zero for flat samples.
    # Keep nir_core's established deterministic behavior for those explicit
    # edge cases rather than allowing NaN/Inf into a production pipeline.
    if np.ptp(resolved_reference) == 0.0 or np.any(array.std(axis=1) == 0.0):
        return native_msc(X, reference=resolved_reference)

    transformer = MultiplicativeScatterCorrection(reference=resolved_reference)
    result = transformer.fit_transform(array)
    if not np.isfinite(result).all():
        raise ValueError("Chemotools MSC produced non-finite output.")
    return _restore_shape(result, was_1d)


def sg_smooth(
    X: np.ndarray,
    window: int = 11,
    order: int = 2,
) -> np.ndarray:
    """Apply Chemotools SG filtering with legacy ``interp`` edge behavior."""
    array, was_1d = _ensure_2d(X)
    transformer = SavitzkyGolayFilter(
        window_length=window,
        polyorder=order,
        mode="interp",
    )
    return _restore_shape(transformer.fit_transform(array), was_1d)


def _resolve_derivative_spacing(
    array: np.ndarray,
    *,
    delta: float,
    spacing_mode: str,
    wv: np.ndarray | None,
) -> float:
    if spacing_mode not in {"index", "wavelength"}:
        raise ValueError(
            f"spacing_mode must be 'index' or 'wavelength', got {spacing_mode!r}."
        )
    if spacing_mode == "wavelength":
        if wv is None:
            raise ValueError(
                "spacing_mode='wavelength' requires a wavelength axis (wv)."
            )
        axis = np.asarray(wv, dtype=float).ravel()
        if axis.shape[0] != array.shape[1]:
            raise ValueError(
                f"wavelength axis length {axis.shape[0]} does not match "
                f"spectrum width {array.shape[1]}."
            )
        differences = np.diff(axis)
        if differences.size == 0:
            raise ValueError("wavelength axis must contain at least two points.")
        if not np.allclose(differences, differences[0], rtol=1e-6, atol=1e-9):
            raise ValueError(
                "spacing_mode='wavelength' requires an equally spaced wavelength "
                "axis; resample to a regular axis first (e.g. nir_align_wavelengths)."
            )
        delta = abs(float(differences[0]))
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError(f"delta must be positive and finite, got {delta!r}.")
    return float(delta)


def sg_derivative(
    X: np.ndarray,
    window: int = 11,
    order: int = 2,
    deriv: int = 1,
    delta: float = 1.0,
    spacing_mode: str = "index",
    wv: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a Chemotools SG derivative with nir_core spacing semantics."""
    if deriv < 0:
        raise ValueError(f"deriv must be non-negative, got {deriv}.")
    if deriv > order:
        raise ValueError(f"deriv must be <= order; got deriv={deriv}, order={order}.")
    array, was_1d = _ensure_2d(X)
    resolved_delta = _resolve_derivative_spacing(
        array,
        delta=delta,
        spacing_mode=spacing_mode,
        wv=wv,
    )
    transformer = SavitzkyGolay(
        window_length=window,
        polyorder=order,
        deriv=deriv,
        mode="interp",
    )
    result = transformer.fit_transform(array) / (resolved_delta**deriv)
    return _restore_shape(result, was_1d)


def derivative1(X: np.ndarray, **kwargs) -> np.ndarray:
    """Public-method adapter for the first SG derivative."""
    return sg_derivative(X, deriv=1, **kwargs)


def derivative2(X: np.ndarray, **kwargs) -> np.ndarray:
    """Public-method adapter for the second SG derivative."""
    return sg_derivative(X, deriv=2, **kwargs)


def norris_williams_derivative(
    X: np.ndarray,
    gap: int = 3,
    segment: int = 5,
    deriv: int = 1,
    delta: float = 1.0,
    mode: str = "nearest",
) -> np.ndarray:
    """Apply Chemotools Norris-Williams with physical derivative scaling.

    Chemotools supplies the smoothing/difference kernel.  The final scale
    preserves nir_core's public ``delta`` meaning so changing wavelength units
    changes the derivative magnitude predictably.
    """
    if deriv not in {1, 2}:
        raise ValueError(f"deriv must be 1 or 2, got {deriv}.")
    if gap < 3 or gap % 2 == 0:
        raise ValueError(f"gap must be an odd integer >= 3, got {gap}.")
    if segment < 1 or segment % 2 == 0:
        raise ValueError(f"segment must be a positive odd integer, got {segment}.")
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError(f"delta must be positive and finite, got {delta!r}.")

    array, was_1d = _ensure_2d(X)
    transformer = NorrisWilliams(
        window_length=segment,
        gap_size=gap,
        deriv=deriv,
        mode=mode,
    )
    result = transformer.fit_transform(array)
    if deriv == 1:
        scale = gap / ((gap - 1) * delta)
    else:
        half_gap = (gap - 1) / 2.0
        scale = gap / ((half_gap * delta) ** 2)
    return _restore_shape(result * scale, was_1d)


def norris_derivative1(X: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Public-method adapter for the first Norris-Williams derivative."""
    return norris_williams_derivative(X, deriv=1, **kwargs)


def norris_derivative2(X: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Public-method adapter for the second Norris-Williams derivative."""
    return norris_williams_derivative(X, deriv=2, **kwargs)


def asls(
    X: np.ndarray,
    lambda_: float = 1e5,
    p: float = 0.001,
) -> np.ndarray:
    """Apply Chemotools asymmetric least squares with project defaults."""
    array, was_1d = _ensure_2d(X)
    if array.shape[1] < 3:
        return _restore_shape(array.copy(), was_1d)
    transformer = AsLs(
        lam=lambda_,
        penalty=p,
        nr_iterations=20,
        max_iter_after_warmstart=20,
        solver_type="banded",
        n_jobs=1,
    )
    return _restore_shape(transformer.fit_transform(array), was_1d)


def _arpls_transformer(
    *,
    lambda_: float,
    ratio: float,
    max_iters: int,
) -> ArPls:
    return ArPls(
        lam=lambda_,
        ratio=ratio,
        nr_iterations=max_iters,
        solver_type="banded",
        max_iter_after_warmstart=20,
        n_jobs=1,
    )


def arpls(
    X: np.ndarray,
    lambda_: float = 1e4,
    ratio: float = 0.01,
    max_iters: int = 100,
) -> np.ndarray:
    """Apply Chemotools asymmetrically reweighted penalized least squares."""
    array, was_1d = _ensure_2d(X)
    if array.shape[1] < 3:
        return _restore_shape(array.copy(), was_1d)
    result = _arpls_transformer(
        lambda_=lambda_,
        ratio=ratio,
        max_iters=max_iters,
    ).fit_transform(array)
    if not np.isfinite(result).all():
        raise ValueError("Chemotools ArPLS produced non-finite output.")
    return _restore_shape(result, was_1d)


def fit_arpls_state(
    X: np.ndarray,
    *,
    params: Mapping[str, Any],
    wv: np.ndarray | None,
) -> dict[str, Any]:
    """Fit ArPLS warm-start state using training spectra only."""
    del wv
    array, _ = _ensure_2d(X)
    if array.shape[1] < 3:
        return {"identity": True, "n_features": int(array.shape[1])}
    transformer = _arpls_transformer(
        lambda_=float(params.get("lambda_", 1e4)),
        ratio=float(params.get("ratio", 0.01)),
        max_iters=int(params.get("max_iters", 100)),
    ).fit(array)
    return {"identity": False, "transformer": transformer}


def transform_arpls_state(
    state: Mapping[str, Any],
    X: np.ndarray,
    *,
    params: Mapping[str, Any],
    wv: np.ndarray | None,
) -> np.ndarray:
    """Replay fitted Chemotools ArPLS warm-start state."""
    del params, wv
    array, was_1d = _ensure_2d(X)
    if state["identity"]:
        if array.shape[1] != state["n_features"]:
            raise ValueError("ArPLS feature count does not match fitted state.")
        return _restore_shape(array.copy(), was_1d)
    result = state["transformer"].transform(array)
    if not np.isfinite(result).all():
        raise ValueError("Chemotools ArPLS produced non-finite output.")
    return _restore_shape(result, was_1d)


def rubberband(X: np.ndarray) -> np.ndarray:
    """Apply Chemotools convex-hull rubber-band baseline correction."""
    array, was_1d = _ensure_2d(X)
    result = RubberbandCorrection(n_jobs=1).fit_transform(array)
    if not np.isfinite(result).all():
        raise ValueError(
            "Chemotools rubber-band correction produced non-finite output."
        )
    return _restore_shape(result, was_1d)


def median_filter(
    X: np.ndarray,
    window_length: int = 3,
    mode: str = "nearest",
) -> np.ndarray:
    """Apply Chemotools median filtering."""
    array, was_1d = _ensure_2d(X)
    result = MedianFilter(
        window_length=window_length,
        mode=mode,
        n_jobs=1,
    ).fit_transform(array)
    return _restore_shape(result, was_1d)


def whittaker_smooth(
    X: np.ndarray,
    lambda_: float = 1e4,
) -> np.ndarray:
    """Apply Chemotools Whittaker smoothing with uniform weights."""
    array, was_1d = _ensure_2d(X)
    if array.shape[1] < 3:
        return _restore_shape(array.copy(), was_1d)
    result = WhittakerSmooth(
        lam=lambda_,
        weights=None,
        solver_type="banded",
        n_jobs=1,
    ).fit_transform(array)
    if not np.isfinite(result).all():
        raise ValueError("Chemotools Whittaker smoothing produced non-finite output.")
    return _restore_shape(result, was_1d)


def _regular_axis(
    wv: np.ndarray | None,
    n_features: int,
) -> np.ndarray | None:
    if wv is None:
        return None
    axis = np.asarray(wv, dtype=float).ravel()
    if axis.shape[0] != n_features:
        raise ValueError(
            f"wavelength axis length {axis.shape[0]} does not match "
            f"spectrum width {n_features}."
        )
    if not np.isfinite(axis).all():
        raise ValueError("wavelength axis must contain only finite values.")
    differences = np.diff(axis)
    monotonic = bool(np.all(differences > 0.0) or np.all(differences < 0.0))
    if not monotonic or not np.allclose(
        differences,
        differences[0],
        rtol=1e-6,
        atol=1e-9,
    ):
        raise ValueError(
            "Chemotools polynomial preprocessing requires a regular wavelength axis; "
            "resample with nir_align_wavelengths first."
        )
    return axis.copy()


def detrend(
    X: np.ndarray,
    *,
    wv: np.ndarray | None = None,
) -> np.ndarray:
    """Remove a quadratic trend with Chemotools PolynomialCorrection."""
    array, was_1d = _ensure_2d(X)
    _regular_axis(wv, array.shape[1])
    result = PolynomialCorrection(order=2, indices=None).fit_transform(array)
    if not np.isfinite(result).all():
        raise ValueError("Chemotools detrend produced non-finite output.")
    return _restore_shape(result, was_1d)


def _checked_emsc_transform(
    transformer: ExtendedMultiplicativeScatterCorrection,
    X: np.ndarray,
    *,
    min_multiplicative: float,
) -> np.ndarray:
    WX = (X * transformer.weights_.ravel()).T
    coefficients, *_ = np.linalg.lstsq(transformer.WA_, WX, rcond=None)
    multiplicative = coefficients[transformer.order + 1]
    invalid = np.abs(multiplicative) <= float(min_multiplicative)
    if np.any(invalid):
        rows = np.flatnonzero(invalid).tolist()
        raise ValueError(
            f"EMSC multiplicative coefficient is too close to zero for rows {rows}."
        )
    result = transformer.transform(X)
    if not np.isfinite(result).all():
        raise ValueError("Chemotools EMSC produced non-finite output.")
    return result


def emsc(
    X: np.ndarray,
    reference: np.ndarray | None = None,
    *,
    wv: np.ndarray | None = None,
    polynomial_order: int = 2,
    min_multiplicative: float = 1e-12,
    method: str = "mean",
) -> np.ndarray:
    """Apply Chemotools EMSC on an index-equivalent regular axis."""
    array, was_1d = _ensure_2d(X)
    _regular_axis(wv, array.shape[1])
    transformer = ExtendedMultiplicativeScatterCorrection(
        method=method,
        order=polynomial_order,
        reference=reference,
    ).fit(array)
    result = _checked_emsc_transform(
        transformer,
        array,
        min_multiplicative=min_multiplicative,
    )
    return _restore_shape(result, was_1d)


def fit_emsc_state(
    X: np.ndarray,
    *,
    params: Mapping[str, Any],
    wv: np.ndarray | None,
) -> dict[str, Any]:
    """Fit a Chemotools EMSC transformer on the training boundary."""
    array, _ = _ensure_2d(X)
    axis = _regular_axis(wv, array.shape[1])
    transformer = ExtendedMultiplicativeScatterCorrection(
        method=str(params.get("method", "mean")),
        order=int(params.get("polynomial_order", 2)),
    ).fit(array)
    return {
        "transformer": transformer,
        "wv": axis,
        "min_multiplicative": float(params.get("min_multiplicative", 1e-12)),
    }


def transform_emsc_state(
    state: Mapping[str, Any],
    X: np.ndarray,
    *,
    params: Mapping[str, Any],
    wv: np.ndarray | None,
) -> np.ndarray:
    """Replay fitted EMSC state and enforce its wavelength-axis contract."""
    del params
    array, was_1d = _ensure_2d(X)
    fitted_axis = state["wv"]
    if fitted_axis is not None:
        current_axis = _regular_axis(wv, array.shape[1])
        if current_axis is None or not np.allclose(
            current_axis,
            fitted_axis,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(
                "EMSC transform wavelength axis does not match the fitted axis."
            )
    result = _checked_emsc_transform(
        state["transformer"],
        array,
        min_multiplicative=float(state["min_multiplicative"]),
    )
    return _restore_shape(result, was_1d)


def _airpls_transformer(
    *,
    lambda_: float,
    max_iters: int,
) -> AirPls:
    return AirPls(
        lam=lambda_,
        nr_iterations=max_iters,
        solver_type="banded",
        max_iter_after_warmstart=20,
        n_jobs=1,
    )


def airpls(
    X: np.ndarray,
    lambda_: float = 1e7,
    porder: int = 1,
    max_iters: int = 100,
) -> np.ndarray:
    """Apply Chemotools AirPLS with the stable nir_core parameter names."""
    del porder
    array, was_1d = _ensure_2d(X)
    if array.shape[1] < 3:
        return _restore_shape(array.copy(), was_1d)
    result = _airpls_transformer(
        lambda_=lambda_,
        max_iters=max_iters,
    ).fit_transform(array)
    if not np.isfinite(result).all():
        raise ValueError("Chemotools AirPLS produced non-finite output.")
    return _restore_shape(result, was_1d)


def fit_airpls_state(
    X: np.ndarray,
    *,
    params: Mapping[str, Any],
    wv: np.ndarray | None,
) -> dict[str, Any]:
    """Fit AirPLS warm-start state using training spectra only."""
    del wv
    array, _ = _ensure_2d(X)
    if array.shape[1] < 3:
        return {"identity": True, "n_features": int(array.shape[1])}
    transformer = _airpls_transformer(
        lambda_=float(params.get("lambda_", 1e7)),
        max_iters=int(params.get("max_iters", 100)),
    ).fit(array)
    return {"identity": False, "transformer": transformer}


def transform_airpls_state(
    state: Mapping[str, Any],
    X: np.ndarray,
    *,
    params: Mapping[str, Any],
    wv: np.ndarray | None,
) -> np.ndarray:
    """Replay fitted Chemotools AirPLS warm-start state."""
    del params, wv
    array, was_1d = _ensure_2d(X)
    if state["identity"]:
        if array.shape[1] != state["n_features"]:
            raise ValueError("AirPLS feature count does not match fitted state.")
        return _restore_shape(array.copy(), was_1d)
    result = state["transformer"].transform(array)
    if not np.isfinite(result).all():
        raise ValueError("Chemotools AirPLS produced non-finite output.")
    return _restore_shape(result, was_1d)


def _fit_pareto(X: np.ndarray, *, p: float) -> ParetoScaler:
    array, _ = _ensure_2d(X)
    transformer = ParetoScaler(p=p, with_mean=True, copy=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The scale for .* feature\(s\) is zero.*",
            category=UserWarning,
        )
        transformer.fit(array)
    return transformer


def mean_center(X: np.ndarray) -> np.ndarray:
    """Mean-center with Chemotools ParetoScaler ``p=0``."""
    array, was_1d = _ensure_2d(X)
    transformer = _fit_pareto(array, p=0.0)
    return _restore_shape(transformer.transform(array), was_1d)


def autoscale(X: np.ndarray) -> np.ndarray:
    """Autoscale with Chemotools ParetoScaler ``p=1``."""
    array, was_1d = _ensure_2d(X)
    transformer = _fit_pareto(array, p=1.0)
    return _restore_shape(transformer.transform(array), was_1d)


def normalize(X: np.ndarray, norm: str = "l2") -> np.ndarray:
    """Apply Chemotools row-wise L1/L2 normalization."""
    if norm not in {"l1", "l2"}:
        raise ValueError(
            f"Chemotools normalize supports only 'l1' and 'l2'; got {norm!r}."
        )
    array, was_1d = _ensure_2d(X)
    transformer = NormScaler(l_norm=1 if norm == "l1" else 2)
    return _restore_shape(transformer.fit_transform(array), was_1d)


def fit_mean_center_state(
    X: np.ndarray,
    *,
    params: Mapping[str, Any],
    wv: np.ndarray | None,
) -> ParetoScaler:
    """Fit the Chemotools mean-centering state on training spectra only."""
    del params, wv
    return _fit_pareto(X, p=0.0)


def fit_autoscale_state(
    X: np.ndarray,
    *,
    params: Mapping[str, Any],
    wv: np.ndarray | None,
) -> ParetoScaler:
    """Fit the Chemotools autoscaling state on training spectra only."""
    del params, wv
    return _fit_pareto(X, p=1.0)


def transform_fitted_scaler(
    state: ParetoScaler,
    X: np.ndarray,
    *,
    params: Mapping[str, Any],
    wv: np.ndarray | None,
) -> np.ndarray:
    """Replay one fitted Chemotools scaler without recomputing statistics."""
    del params, wv
    array, was_1d = _ensure_2d(X)
    return _restore_shape(state.transform(array), was_1d)


__all__ = [
    "EXTENDED_IMPLEMENTATION_VERSION",
    "IMPLEMENTATION_VERSION",
    "PROVIDER_NAME",
    "PROVIDER_VERSION",
    "airpls",
    "arpls",
    "asls",
    "autoscale",
    "derivative1",
    "derivative2",
    "detrend",
    "emsc",
    "fit_airpls_state",
    "fit_arpls_state",
    "fit_autoscale_state",
    "fit_emsc_state",
    "fit_mean_center_state",
    "mean_center",
    "median_filter",
    "msc",
    "normalize",
    "norris_derivative1",
    "norris_derivative2",
    "norris_williams_derivative",
    "rnv",
    "rubberband",
    "sg_derivative",
    "sg_smooth",
    "snv",
    "transform_airpls_state",
    "transform_arpls_state",
    "transform_emsc_state",
    "transform_fitted_scaler",
    "whittaker_smooth",
]
