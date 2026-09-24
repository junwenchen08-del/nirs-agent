"""Authoritative catalog for ordinary NIR preprocessing methods.

The catalog contains stable public method identifiers and the metadata needed
by pipeline validation, agent-facing discovery tools, documentation, and model
artifact provenance.  The initial catalog deliberately keeps every method on
the existing native implementation; provider changes are introduced only
after numerical compatibility tests pass.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from nir_core.preprocess.baseline import airpls, asls, detrend
from nir_core.preprocess.despike import despike
from nir_core.preprocess.providers import chemotools as chemotools_provider
from nir_core.preprocess.scaling import autoscale, mean_center, normalize
from nir_core.preprocess.scatter import emsc, msc, robust_snv, snv
from nir_core.preprocess.smoothing import (
    norris_williams_derivative,
    sg_derivative,
    sg_smooth,
)

CATALOG_VERSION = "2.0"
NATIVE_IMPLEMENTATION_VERSION = "native-v1"

_UNSET = object()

ParameterKind = Literal["integer", "number", "string", "boolean"]
AutoLevel = Literal["default", "conditional", "explicit_only", "disabled"]
MethodStatus = Literal["stable", "experimental", "unavailable", "deprecated"]
CostLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ParameterSpec:
    """Validated public parameter metadata for one preprocessing method."""

    kind: ParameterKind
    description_zh: str
    default: Any = _UNSET
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] = ()
    odd: bool = False
    display: bool = True

    @property
    def has_default(self) -> bool:
        """Whether a public default is defined."""
        return self.default is not _UNSET

    def as_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe representation."""
        result: dict[str, Any] = {
            "type": self.kind,
            "description_zh": self.description_zh,
        }
        if self.has_default:
            result["default"] = self.default
        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.maximum is not None:
            result["maximum"] = self.maximum
        if self.choices:
            result["choices"] = list(self.choices)
        if self.odd:
            result["odd"] = True
        return result


@dataclass(frozen=True)
class ProviderImplementationSpec:
    """One versioned implementation option for a public method."""

    provider: str
    implementation: Any = field(repr=False, compare=False)
    provider_class: str
    provider_version: str | None
    implementation_version: str
    compatibility: Literal["verified", "native", "experimental"] = "verified"
    fit_state: Any = field(default=None, repr=False, compare=False)
    transform_state: Any = field(default=None, repr=False, compare=False)

    def as_dict(self, *, is_default: bool) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "provider_class": self.provider_class,
            "provider_version": self.provider_version,
            "implementation_version": self.implementation_version,
            "compatibility": self.compatibility,
            "is_default": is_default,
            "has_fitted_state": self.fit_state is not None,
        }


@dataclass(frozen=True)
class PreprocessingMethodSpec:
    """Single source of truth for one public preprocessing method."""

    method_id: str
    implementation: Any = field(repr=False, compare=False)
    display_name_zh: str
    summary_zh: str
    category: str
    stage_order: int
    provider: str = "native"
    preferred_provider: str = "native"
    provider_class: str | None = None
    provider_version: str | None = None
    implementation_version: str = NATIVE_IMPLEMENTATION_VERSION
    fit_state: Any = field(default=None, repr=False, compare=False)
    transform_state: Any = field(default=None, repr=False, compare=False)
    alternative_providers: Mapping[str, ProviderImplementationSpec] = field(
        default_factory=dict
    )
    parameter_schema: Mapping[str, ParameterSpec] = field(default_factory=dict)
    search_space: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    stateful: bool = False
    requires_wavelengths: bool = False
    requires_regular_axis: bool = False
    requires_y: bool = False
    requires_reference: bool = False
    requires_interferences: bool = False
    changes_axis: bool = False
    stochastic: bool = False
    exclusive_group: str | None = None
    recommended_when: tuple[str, ...] = ()
    avoid_when: tuple[str, ...] = ()
    auto_level: AutoLevel = "explicit_only"
    cost_level: CostLevel = "low"
    status: MethodStatus = "stable"
    provider_selector: Any = field(default=None, repr=False, compare=False)

    def compact_dict(self) -> dict[str, Any]:
        """Return the bounded record intended for method-list responses."""
        return {
            "method_id": self.method_id,
            "display_name_zh": self.display_name_zh,
            "summary_zh": self.summary_zh,
            "category": self.category,
            "provider": self.provider,
            "preferred_provider": self.preferred_provider,
            "stateful": self.stateful,
            "requires_wavelengths": self.requires_wavelengths,
            "auto_level": self.auto_level,
            "cost_level": self.cost_level,
            "status": self.status,
        }

    def detailed_dict(self) -> dict[str, Any]:
        """Return the complete JSON-safe catalog record."""
        providers = [
            ProviderImplementationSpec(
                provider=self.provider,
                implementation=self.implementation,
                provider_class=self.provider_class or "",
                provider_version=self.provider_version,
                implementation_version=self.implementation_version,
                compatibility="native" if self.provider == "native" else "verified",
                fit_state=self.fit_state,
                transform_state=self.transform_state,
            ).as_dict(is_default=self.preferred_provider == self.provider)
        ]
        providers.extend(
            self.alternative_providers[name].as_dict(
                is_default=self.preferred_provider == name
            )
            for name in sorted(self.alternative_providers)
        )
        return {
            **self.compact_dict(),
            "provider_class": self.provider_class,
            "provider_version": self.provider_version,
            "implementation_version": self.implementation_version,
            "available_providers": providers,
            "stage_order": self.stage_order,
            "parameter_schema": {
                name: parameter.as_dict()
                for name, parameter in sorted(self.parameter_schema.items())
            },
            "search_space": {
                name: list(values) for name, values in sorted(self.search_space.items())
            },
            "requires_regular_axis": self.requires_regular_axis,
            "requires_y": self.requires_y,
            "requires_reference": self.requires_reference,
            "requires_interferences": self.requires_interferences,
            "changes_axis": self.changes_axis,
            "stochastic": self.stochastic,
            "exclusive_group": self.exclusive_group,
            "recommended_when": list(self.recommended_when),
            "avoid_when": list(self.avoid_when),
        }


def _derivative1(X: np.ndarray, **kwargs: Any) -> np.ndarray:
    return sg_derivative(X, deriv=1, **kwargs)


def _derivative2(X: np.ndarray, **kwargs: Any) -> np.ndarray:
    return sg_derivative(X, deriv=2, **kwargs)


def _norris_derivative1(X: np.ndarray, **kwargs: Any) -> np.ndarray:
    return norris_williams_derivative(X, deriv=1, **kwargs)


def _norris_derivative2(X: np.ndarray, **kwargs: Any) -> np.ndarray:
    return norris_williams_derivative(X, deriv=2, **kwargs)


def _params(**items: ParameterSpec) -> Mapping[str, ParameterSpec]:
    return MappingProxyType(dict(items))


def _space(**items: tuple[Any, ...]) -> Mapping[str, tuple[Any, ...]]:
    return MappingProxyType(dict(items))


def _alternative_providers(
    **items: ProviderImplementationSpec,
) -> Mapping[str, ProviderImplementationSpec]:
    return MappingProxyType(dict(items))


def _chemotools_option(
    implementation: Any,
    provider_class: str,
    *,
    fit_state: Any = None,
    transform_state: Any = None,
    implementation_version: str | None = None,
) -> ProviderImplementationSpec:
    return ProviderImplementationSpec(
        provider=chemotools_provider.PROVIDER_NAME,
        implementation=implementation,
        provider_class=provider_class,
        provider_version=chemotools_provider.PROVIDER_VERSION,
        implementation_version=(
            implementation_version or chemotools_provider.IMPLEMENTATION_VERSION
        ),
        compatibility="verified",
        fit_state=fit_state,
        transform_state=transform_state,
    )


def _normalize_provider(params: Mapping[str, Any]) -> str:
    return "native" if params.get("norm", "l2") == "max" else "chemotools"


def _number(
    description: str,
    default: float,
    minimum: float,
    maximum: float,
) -> ParameterSpec:
    return ParameterSpec(
        kind="number",
        description_zh=description,
        default=default,
        minimum=minimum,
        maximum=maximum,
    )


def _integer(
    description: str,
    default: int,
    minimum: int,
    maximum: int,
    *,
    odd: bool = False,
) -> ParameterSpec:
    return ParameterSpec(
        kind="integer",
        description_zh=description,
        default=default,
        minimum=minimum,
        maximum=maximum,
        odd=odd,
    )


_METHOD_SPECS = (
    PreprocessingMethodSpec(
        method_id="snv",
        implementation=snv,
        provider_class="nir_core.preprocess.scatter.snv",
        preferred_provider="chemotools",
        alternative_providers=_alternative_providers(
            chemotools=_chemotools_option(
                chemotools_provider.snv,
                "chemotools.scatter.StandardNormalVariate",
            )
        ),
        display_name_zh="SNV",
        summary_zh="逐条光谱执行标准正态变量变换，降低加性和乘性散射影响。",
        category="scatter",
        stage_order=1,
        exclusive_group="scatter",
        recommended_when=("additive_scatter", "multiplicative_scatter"),
        auto_level="default",
    ),
    PreprocessingMethodSpec(
        method_id="robust_snv",
        implementation=robust_snv,
        provider_class="nir_core.preprocess.scatter.robust_snv",
        display_name_zh="稳健SNV",
        summary_zh="使用中位数和 MAD 的稳健 SNV，降低异常点和重尾噪声影响。",
        category="scatter",
        stage_order=1,
        parameter_schema=_params(
            consistency=_number("MAD 一致性缩放系数", 1.4826, 1e-12, 10.0)
        ),
        exclusive_group="scatter",
        recommended_when=("multiplicative_scatter", "heavy_tailed_rows"),
        auto_level="explicit_only",
    ),
    PreprocessingMethodSpec(
        method_id="rnv",
        implementation=chemotools_provider.rnv,
        provider="chemotools",
        preferred_provider="chemotools",
        provider_class="chemotools.scatter.RobustNormalVariate",
        provider_version=chemotools_provider.PROVIDER_VERSION,
        implementation_version=chemotools_provider.EXTENDED_IMPLEMENTATION_VERSION,
        display_name_zh="RNV稳健正态变量",
        summary_zh="按分位数选取稳健子集估计尺度，降低重尾噪声和异常波段影响。",
        category="scatter",
        stage_order=1,
        parameter_schema=_params(
            percentile=_number("稳健尺度估计分位数", 25.0, 0.0, 100.0),
            epsilon=_number("防止除零的稳定项", 1e-10, 1e-15, 1.0),
        ),
        exclusive_group="scatter",
        recommended_when=("multiplicative_scatter", "heavy_tailed_rows"),
        auto_level="conditional",
    ),
    PreprocessingMethodSpec(
        method_id="msc",
        implementation=msc,
        provider_class="nir_core.preprocess.scatter.msc",
        preferred_provider="chemotools",
        alternative_providers=_alternative_providers(
            chemotools=_chemotools_option(
                chemotools_provider.msc,
                "chemotools.scatter.MultiplicativeScatterCorrection",
            )
        ),
        display_name_zh="MSC",
        summary_zh="使用训练参考谱拟合并校正加性和乘性散射。",
        category="scatter",
        stage_order=1,
        stateful=True,
        exclusive_group="scatter",
        recommended_when=("additive_scatter", "multiplicative_scatter"),
        auto_level="default",
    ),
    PreprocessingMethodSpec(
        method_id="emsc",
        implementation=emsc,
        provider_class="nir_core.preprocess.scatter.emsc",
        preferred_provider="chemotools",
        alternative_providers=_alternative_providers(
            chemotools=_chemotools_option(
                chemotools_provider.emsc,
                "chemotools.scatter.ExtendedMultiplicativeScatterCorrection",
                fit_state=chemotools_provider.fit_emsc_state,
                transform_state=chemotools_provider.transform_emsc_state,
                implementation_version=(
                    chemotools_provider.EXTENDED_IMPLEMENTATION_VERSION
                ),
            )
        ),
        display_name_zh="EMSC",
        summary_zh="在训练参考谱基础上同时建模散射和低阶基线项。",
        category="scatter",
        stage_order=1,
        parameter_schema=_params(
            polynomial_order=_integer("多项式基线阶数", 2, 0, 3),
            min_multiplicative=_number("允许的最小乘性系数绝对值", 1e-12, 1e-15, 1.0),
            method=ParameterSpec(
                kind="string",
                description_zh="未提供参考谱时的训练参考统计量",
                default="mean",
                choices=("mean", "median"),
            ),
        ),
        search_space=_space(polynomial_order=(0, 1, 2)),
        stateful=True,
        requires_wavelengths=False,
        requires_regular_axis=True,
        exclusive_group="scatter",
        recommended_when=("additive_scatter", "multiplicative_scatter"),
        auto_level="explicit_only",
        cost_level="medium",
    ),
    PreprocessingMethodSpec(
        method_id="despike",
        implementation=despike,
        provider_class="nir_core.preprocess.despike.despike",
        display_name_zh="去尖峰",
        summary_zh="使用局部稳健残差检测并替换孤立仪器尖峰。",
        category="despike",
        stage_order=-1,
        parameter_schema=_params(
            window=_integer("局部中位数窗口", 5, 3, 51, odd=True),
            z_threshold=_number("稳健尖峰阈值", 6.0, 2.0, 20.0),
        ),
        recommended_when=("isolated_spikes",),
        exclusive_group="spike_correction",
        auto_level="explicit_only",
    ),
    PreprocessingMethodSpec(
        method_id="sg_smooth",
        implementation=sg_smooth,
        provider_class="nir_core.preprocess.smoothing.sg_smooth",
        preferred_provider="chemotools",
        alternative_providers=_alternative_providers(
            chemotools=_chemotools_option(
                chemotools_provider.sg_smooth,
                "chemotools.smooth.SavitzkyGolayFilter",
            )
        ),
        display_name_zh="SG平滑",
        summary_zh="使用 Savitzky-Golay 多项式窗口降低高频噪声。",
        category="smooth",
        stage_order=1,
        parameter_schema=_params(
            window=_integer("Savitzky-Golay 窗口", 11, 5, 15, odd=True),
            order=_integer("局部多项式阶数", 2, 1, 4),
        ),
        search_space=_space(window=(7, 11, 15), order=(2, 3)),
        recommended_when=("high_frequency_noise",),
        auto_level="default",
    ),
    PreprocessingMethodSpec(
        method_id="whittaker_smooth",
        implementation=chemotools_provider.whittaker_smooth,
        provider="chemotools",
        preferred_provider="chemotools",
        provider_class="chemotools.smooth.WhittakerSmooth",
        provider_version=chemotools_provider.PROVIDER_VERSION,
        implementation_version=chemotools_provider.EXTENDED_IMPLEMENTATION_VERSION,
        display_name_zh="Whittaker平滑",
        summary_zh="用惩罚最小二乘平衡平滑程度与原始光谱保真度。",
        category="smooth",
        stage_order=1,
        parameter_schema=_params(
            lambda_=_number("平滑惩罚强度", 1e4, 1.0, 1e10),
        ),
        search_space=_space(lambda_=(1e2, 1e3, 1e4, 1e5)),
        recommended_when=("high_frequency_noise",),
        auto_level="conditional",
        cost_level="medium",
    ),
    PreprocessingMethodSpec(
        method_id="median_filter",
        implementation=chemotools_provider.median_filter,
        provider="chemotools",
        preferred_provider="chemotools",
        provider_class="chemotools.smooth.MedianFilter",
        provider_version=chemotools_provider.PROVIDER_VERSION,
        implementation_version=chemotools_provider.EXTENDED_IMPLEMENTATION_VERSION,
        display_name_zh="中值滤波",
        summary_zh="用局部中值抑制密集脉冲噪声，可能削弱窄峰。",
        category="smooth",
        stage_order=1,
        parameter_schema=_params(
            window_length=_integer("奇数滤波窗口", 3, 1, 51, odd=True),
            mode=ParameterSpec(
                kind="string",
                description_zh="边界扩展方式",
                default="nearest",
                choices=(
                    "reflect",
                    "constant",
                    "nearest",
                    "mirror",
                    "wrap",
                    "grid-constant",
                    "grid-mirror",
                    "grid-wrap",
                ),
            ),
        ),
        exclusive_group="spike_correction",
        recommended_when=("dense_impulse_noise",),
        avoid_when=("narrow_analytical_peaks",),
        auto_level="explicit_only",
    ),
    PreprocessingMethodSpec(
        method_id="derivative1",
        implementation=_derivative1,
        provider_class="nir_core.preprocess.smoothing.sg_derivative",
        preferred_provider="chemotools",
        alternative_providers=_alternative_providers(
            chemotools=_chemotools_option(
                chemotools_provider.derivative1,
                "chemotools.derivative.SavitzkyGolay",
            )
        ),
        display_name_zh="一阶导数",
        summary_zh="使用 Savitzky-Golay 一阶导数削弱基线并增强重叠峰差异。",
        category="derivative",
        stage_order=1,
        parameter_schema=_params(
            window=_integer("Savitzky-Golay 窗口", 11, 5, 15, odd=True),
            order=_integer("局部多项式阶数", 2, 1, 4),
            delta=_number("相邻点间隔", 1.0, 1e-12, 1e12),
            spacing_mode=ParameterSpec(
                kind="string",
                description_zh="按列索引或物理波长间隔求导",
                default="index",
                choices=("index", "wavelength"),
            ),
        ),
        search_space=_space(window=(7, 11, 15), order=(2, 3)),
        exclusive_group="derivative",
        recommended_when=("broad_overlapping_bands", "baseline_drift"),
        avoid_when=("low_signal_to_noise",),
        auto_level="default",
    ),
    PreprocessingMethodSpec(
        method_id="derivative2",
        implementation=_derivative2,
        provider_class="nir_core.preprocess.smoothing.sg_derivative",
        preferred_provider="chemotools",
        alternative_providers=_alternative_providers(
            chemotools=_chemotools_option(
                chemotools_provider.derivative2,
                "chemotools.derivative.SavitzkyGolay",
            )
        ),
        display_name_zh="二阶导数",
        summary_zh="使用 Savitzky-Golay 二阶导数增强曲率信息，噪声较大时慎用。",
        category="derivative",
        stage_order=1,
        parameter_schema=_params(
            window=_integer("Savitzky-Golay 窗口", 11, 5, 15, odd=True),
            order=_integer("局部多项式阶数", 2, 2, 4),
            delta=_number("相邻点间隔", 1.0, 1e-12, 1e12),
            spacing_mode=ParameterSpec(
                kind="string",
                description_zh="按列索引或物理波长间隔求导",
                default="index",
                choices=("index", "wavelength"),
            ),
        ),
        exclusive_group="derivative",
        recommended_when=("broad_overlapping_bands",),
        avoid_when=("low_signal_to_noise",),
        auto_level="explicit_only",
    ),
    PreprocessingMethodSpec(
        method_id="norris_derivative1",
        implementation=_norris_derivative1,
        provider_class="nir_core.preprocess.smoothing.norris_williams_derivative",
        preferred_provider="chemotools",
        alternative_providers=_alternative_providers(
            chemotools=_chemotools_option(
                chemotools_provider.norris_derivative1,
                "chemotools.derivative.NorrisWilliams",
                implementation_version=(
                    chemotools_provider.EXTENDED_IMPLEMENTATION_VERSION
                ),
            )
        ),
        display_name_zh="Norris-Williams一阶导数",
        summary_zh="使用 Norris-Williams 间隔算法计算一阶导数。",
        category="derivative",
        stage_order=1,
        parameter_schema=_params(
            gap=_integer("奇数差分间隔", 3, 3, 49, odd=True),
            segment=_integer("奇数平滑窗口", 5, 1, 51, odd=True),
            delta=_number("相邻点间隔", 1.0, 1e-12, 1e12),
            mode=ParameterSpec(
                kind="string",
                description_zh="边界扩展方式",
                default="nearest",
                choices=("nearest", "constant", "reflect", "wrap", "mirror"),
            ),
        ),
        requires_regular_axis=True,
        exclusive_group="derivative",
        auto_level="explicit_only",
    ),
    PreprocessingMethodSpec(
        method_id="norris_derivative2",
        implementation=_norris_derivative2,
        provider_class="nir_core.preprocess.smoothing.norris_williams_derivative",
        preferred_provider="chemotools",
        alternative_providers=_alternative_providers(
            chemotools=_chemotools_option(
                chemotools_provider.norris_derivative2,
                "chemotools.derivative.NorrisWilliams",
                implementation_version=(
                    chemotools_provider.EXTENDED_IMPLEMENTATION_VERSION
                ),
            )
        ),
        display_name_zh="Norris-Williams二阶导数",
        summary_zh="使用 Norris-Williams 间隔算法计算二阶导数。",
        category="derivative",
        stage_order=1,
        parameter_schema=_params(
            gap=_integer("奇数差分间隔", 3, 3, 49, odd=True),
            segment=_integer("奇数平滑窗口", 5, 1, 51, odd=True),
            delta=_number("相邻点间隔", 1.0, 1e-12, 1e12),
            mode=ParameterSpec(
                kind="string",
                description_zh="边界扩展方式",
                default="nearest",
                choices=("nearest", "constant", "reflect", "wrap", "mirror"),
            ),
        ),
        requires_regular_axis=True,
        exclusive_group="derivative",
        avoid_when=("low_signal_to_noise",),
        auto_level="explicit_only",
    ),
    PreprocessingMethodSpec(
        method_id="airpls",
        implementation=airpls,
        provider_class="nir_core.preprocess.baseline.airpls",
        preferred_provider="chemotools",
        alternative_providers=_alternative_providers(
            chemotools=_chemotools_option(
                chemotools_provider.airpls,
                "chemotools.baseline.AirPls",
                fit_state=chemotools_provider.fit_airpls_state,
                transform_state=chemotools_provider.transform_airpls_state,
                implementation_version=(
                    chemotools_provider.EXTENDED_IMPLEMENTATION_VERSION
                ),
            )
        ),
        display_name_zh="airPLS基线校正",
        summary_zh="使用自适应迭代惩罚最小二乘估计并去除平滑基线。",
        category="baseline",
        stage_order=0,
        parameter_schema=_params(
            lambda_=_number("平滑惩罚强度", 1e7, 1e3, 1e8),
            porder=_integer("差分惩罚阶数", 1, 1, 3),
            max_iters=_integer("最大迭代次数", 100, 1, 1000),
        ),
        search_space=_space(lambda_=(1e4, 1e5, 1e6, 1e7)),
        stateful=True,
        fit_state=chemotools_provider.fit_arpls_state,
        transform_state=chemotools_provider.transform_arpls_state,
        exclusive_group="iterative_baseline",
        recommended_when=("baseline_drift",),
        auto_level="conditional",
        cost_level="medium",
    ),
    PreprocessingMethodSpec(
        method_id="arpls",
        implementation=chemotools_provider.arpls,
        provider="chemotools",
        preferred_provider="chemotools",
        provider_class="chemotools.baseline.ArPls",
        provider_version=chemotools_provider.PROVIDER_VERSION,
        implementation_version=chemotools_provider.EXTENDED_IMPLEMENTATION_VERSION,
        display_name_zh="ArPLS基线校正",
        summary_zh="自适应重加权惩罚最小二乘基线校正，不依赖人工非对称参数。",
        category="baseline",
        stage_order=0,
        parameter_schema=_params(
            lambda_=_number("平滑惩罚强度", 1e4, 1.0, 1e10),
            ratio=_number("迭代收敛阈值", 0.01, 1e-6, 1.0),
            max_iters=_integer("最大迭代次数", 100, 1, 1000),
        ),
        search_space=_space(
            lambda_=(1e3, 1e4, 1e5, 1e6),
            ratio=(0.001, 0.01, 0.05),
        ),
        stateful=True,
        exclusive_group="iterative_baseline",
        recommended_when=("baseline_drift",),
        auto_level="conditional",
        cost_level="medium",
    ),
    PreprocessingMethodSpec(
        method_id="asls",
        implementation=asls,
        provider_class="nir_core.preprocess.baseline.asls",
        preferred_provider="chemotools",
        alternative_providers=_alternative_providers(
            chemotools=_chemotools_option(
                chemotools_provider.asls,
                "chemotools.baseline.AsLs",
            )
        ),
        display_name_zh="asLS基线校正",
        summary_zh="使用非对称最小二乘估计并去除平滑基线。",
        category="baseline",
        stage_order=0,
        parameter_schema=_params(
            lambda_=_number("平滑惩罚强度", 1e5, 1e3, 1e8),
            p=_number("非对称权重", 0.001, 0.001, 0.999),
        ),
        search_space=_space(
            lambda_=(1e4, 1e5, 1e6),
            p=(0.001, 0.01, 0.05),
        ),
        exclusive_group="iterative_baseline",
        recommended_when=("baseline_drift",),
        auto_level="conditional",
        cost_level="medium",
    ),
    PreprocessingMethodSpec(
        method_id="detrend",
        implementation=detrend,
        provider_class="nir_core.preprocess.baseline.detrend",
        preferred_provider="chemotools",
        alternative_providers=_alternative_providers(
            chemotools=_chemotools_option(
                chemotools_provider.detrend,
                "chemotools.baseline.PolynomialCorrection",
                implementation_version=(
                    chemotools_provider.EXTENDED_IMPLEMENTATION_VERSION
                ),
            )
        ),
        display_name_zh="去趋势",
        summary_zh="逐条光谱拟合并去除二次多项式趋势。",
        category="baseline",
        stage_order=0,
        requires_regular_axis=True,
        recommended_when=("baseline_drift",),
        auto_level="conditional",
    ),
    PreprocessingMethodSpec(
        method_id="rubberband",
        implementation=chemotools_provider.rubberband,
        provider="chemotools",
        preferred_provider="chemotools",
        provider_class="chemotools.baseline.RubberbandCorrection",
        provider_version=chemotools_provider.PROVIDER_VERSION,
        implementation_version=chemotools_provider.EXTENDED_IMPLEMENTATION_VERSION,
        display_name_zh="橡皮筋基线校正",
        summary_zh="用光谱下凸包构造橡皮筋基线，适合缓慢变化的荧光或背景隆起。",
        category="baseline",
        stage_order=0,
        exclusive_group="iterative_baseline",
        recommended_when=("rubberband_like_baseline",),
        avoid_when=("broad_baseline_humps",),
        auto_level="explicit_only",
    ),
    PreprocessingMethodSpec(
        method_id="mean_center",
        implementation=mean_center,
        provider_class="nir_core.preprocess.scaling.mean_center",
        preferred_provider="chemotools",
        alternative_providers=_alternative_providers(
            chemotools=_chemotools_option(
                chemotools_provider.mean_center,
                "chemotools.scale.ParetoScaler",
                fit_state=chemotools_provider.fit_mean_center_state,
                transform_state=chemotools_provider.transform_fitted_scaler,
                implementation_version=(
                    chemotools_provider.EXTENDED_IMPLEMENTATION_VERSION
                ),
            )
        ),
        display_name_zh="均值中心化",
        summary_zh="使用训练数据的逐波长均值进行中心化。",
        category="scaling",
        stage_order=2,
        stateful=True,
        exclusive_group="scaling",
        auto_level="default",
    ),
    PreprocessingMethodSpec(
        method_id="autoscale",
        implementation=autoscale,
        provider_class="nir_core.preprocess.scaling.autoscale",
        preferred_provider="chemotools",
        alternative_providers=_alternative_providers(
            chemotools=_chemotools_option(
                chemotools_provider.autoscale,
                "chemotools.scale.ParetoScaler",
                fit_state=chemotools_provider.fit_autoscale_state,
                transform_state=chemotools_provider.transform_fitted_scaler,
                implementation_version=(
                    chemotools_provider.EXTENDED_IMPLEMENTATION_VERSION
                ),
            )
        ),
        display_name_zh="自动缩放",
        summary_zh="使用训练均值和标准差执行单位方差缩放。",
        category="scaling",
        stage_order=2,
        stateful=True,
        exclusive_group="scaling",
        auto_level="default",
    ),
    PreprocessingMethodSpec(
        method_id="normalize",
        implementation=normalize,
        provider_class="nir_core.preprocess.scaling.normalize",
        preferred_provider="chemotools",
        alternative_providers=_alternative_providers(
            chemotools=_chemotools_option(
                chemotools_provider.normalize,
                "chemotools.scale.NormScaler",
                implementation_version=(
                    chemotools_provider.EXTENDED_IMPLEMENTATION_VERSION
                ),
            )
        ),
        display_name_zh="归一化",
        summary_zh="将每条光谱缩放到指定向量范数。",
        category="scaling",
        stage_order=2,
        parameter_schema=_params(
            norm=ParameterSpec(
                kind="string",
                description_zh="向量范数",
                default="l2",
                choices=("l1", "l2", "max"),
            )
        ),
        exclusive_group="scaling",
        auto_level="conditional",
        provider_selector=_normalize_provider,
    ),
)


def _build_registry() -> Mapping[str, PreprocessingMethodSpec]:
    registry: dict[str, PreprocessingMethodSpec] = {}
    for spec in _METHOD_SPECS:
        if spec.method_id in registry:
            raise RuntimeError(f"Duplicate preprocessing method id: {spec.method_id}")
        available_providers = {spec.provider, *spec.alternative_providers}
        if spec.preferred_provider not in available_providers:
            raise RuntimeError(
                f"Preferred provider {spec.preferred_provider!r} is not registered "
                f"for preprocessing method {spec.method_id!r}"
            )
        registry[spec.method_id] = spec
    return MappingProxyType(registry)


METHOD_REGISTRY = _build_registry()


def get_method_spec(method_id: str) -> PreprocessingMethodSpec:
    """Return one method specification or raise a stable lookup error."""
    try:
        return METHOD_REGISTRY[method_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown preprocessing method: {method_id!r}. "
            f"Available: {sorted(METHOD_REGISTRY)}"
        ) from exc


def list_methods(
    *,
    categories: set[str] | None = None,
    auto_levels: set[str] | None = None,
    statuses: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return compact, deterministically ordered method records."""
    result: list[dict[str, Any]] = []
    for method_id in sorted(METHOD_REGISTRY):
        spec = METHOD_REGISTRY[method_id]
        if categories is not None and spec.category not in categories:
            continue
        if auto_levels is not None and spec.auto_level not in auto_levels:
            continue
        if statuses is not None and spec.status not in statuses:
            continue
        result.append(spec.compact_dict())
    return result


def describe_method(method_id: str) -> dict[str, Any]:
    """Return the complete public record for one method."""
    return get_method_spec(method_id).detailed_dict()


def get_provider_implementation(
    method_id: str,
    provider: str | None = None,
) -> ProviderImplementationSpec:
    """Resolve one explicit provider without runtime try/fallback behavior."""
    method = get_method_spec(method_id)
    selected = provider or method.provider
    if selected == method.provider:
        return ProviderImplementationSpec(
            provider=method.provider,
            implementation=method.implementation,
            provider_class=method.provider_class or "",
            provider_version=method.provider_version,
            implementation_version=method.implementation_version,
            compatibility="native" if method.provider == "native" else "verified",
            fit_state=method.fit_state,
            transform_state=method.transform_state,
        )
    try:
        return method.alternative_providers[selected]
    except KeyError as exc:
        available = [method.provider, *sorted(method.alternative_providers)]
        raise KeyError(
            f"Provider {selected!r} is not available for method {method_id!r}. "
            f"Available: {available}"
        ) from exc


def resolve_preferred_provider(
    method_id: str,
    params: Mapping[str, Any],
) -> str:
    """Resolve the catalog provider once, including declared parameter rules."""
    method = get_method_spec(method_id)
    selected = (
        method.provider_selector(params)
        if method.provider_selector is not None
        else method.preferred_provider
    )
    get_provider_implementation(method_id, selected)
    return selected


def _validate_parameter(name: str, value: Any, spec: ParameterSpec) -> str | None:
    if spec.kind == "integer":
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            return f"参数 {name}={value!r} 必须是整数"
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            return f"参数 {name}={value!r} 必须是整数"
        if spec.odd and int(numeric) % 2 == 0:
            return f"参数 {name} 必须是奇数 (当前值 {int(numeric)})"
    elif spec.kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            return f"参数 {name}={value!r} 必须是有限数值"
        numeric = float(value)
        if not math.isfinite(numeric):
            return f"参数 {name}={value!r} 必须是有限数值"
    elif spec.kind == "string":
        if not isinstance(value, str):
            return f"参数 {name}={value!r} 必须是字符串"
        numeric = None
    elif spec.kind == "boolean":
        if not isinstance(value, bool):
            return f"参数 {name}={value!r} 必须是布尔值"
        numeric = None
    else:  # pragma: no cover - catalog construction owns the finite kind set
        return f"参数 {name} 使用了未知类型 {spec.kind!r}"

    if spec.choices and value not in spec.choices:
        return f"参数 {name}={value!r} 不在可选值 {list(spec.choices)!r} 中"
    if spec.kind in {"integer", "number"}:
        if spec.minimum is not None and numeric < spec.minimum:
            return f"参数 {name}={value!r} 超出范围 [{spec.minimum}, {spec.maximum}]"
        if spec.maximum is not None and numeric > spec.maximum:
            return f"参数 {name}={value!r} 超出范围 [{spec.minimum}, {spec.maximum}]"
    return None


def validate_method_params(
    method_id: str,
    params: Mapping[str, Any],
) -> tuple[bool, str]:
    """Validate public parameters against the authoritative method schema."""
    try:
        method = get_method_spec(method_id)
    except KeyError as exc:
        return False, str(exc)

    unknown = sorted(set(params) - set(method.parameter_schema))
    if unknown:
        return (
            False,
            (
                f"方法 {method_id!r} 包含未知参数 {unknown}，"
                f"允许参数: {sorted(method.parameter_schema)}"
            ),
        )

    for name, value in params.items():
        reason = _validate_parameter(name, value, method.parameter_schema[name])
        if reason is not None:
            return False, f"{reason} (方法 {method_id!r})"

    if method_id in {"sg_smooth", "derivative1", "derivative2"}:
        window = params.get("window", method.parameter_schema["window"].default)
        order = params.get("order", method.parameter_schema["order"].default)
        if int(window) <= int(order):
            return False, f"window ({window}) 必须大于 order ({order})"

    return True, ""


def catalog_snapshot() -> dict[str, Any]:
    """Return the complete deterministic, JSON-safe catalog snapshot."""
    return {
        "catalog_version": CATALOG_VERSION,
        "methods": [
            METHOD_REGISTRY[method_id].detailed_dict()
            for method_id in sorted(METHOD_REGISTRY)
        ],
    }


def catalog_hash() -> str:
    """Return a stable SHA-256 for the public catalog semantics."""
    payload = json.dumps(
        catalog_snapshot(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_exclusive_groups() -> list[set[str]]:
    """Build the legacy exclusivity view from catalog group identifiers."""
    groups: dict[str, set[str]] = {}
    for spec in METHOD_REGISTRY.values():
        if spec.exclusive_group is not None:
            groups.setdefault(spec.exclusive_group, set()).add(spec.method_id)
    return [groups[name] for name in sorted(groups)]


__all__ = [
    "CATALOG_VERSION",
    "METHOD_REGISTRY",
    "ParameterSpec",
    "PreprocessingMethodSpec",
    "ProviderImplementationSpec",
    "build_exclusive_groups",
    "catalog_hash",
    "catalog_snapshot",
    "describe_method",
    "get_method_spec",
    "get_provider_implementation",
    "list_methods",
    "resolve_preferred_provider",
    "validate_method_params",
]
