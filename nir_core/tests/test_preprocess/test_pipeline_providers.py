"""Provider binding and legacy-artifact compatibility tests."""

from __future__ import annotations

import joblib
import numpy as np
import pytest

from nir_core.models import PreprocessingStep
from nir_core.preprocess.pipeline import PreprocessingPipeline


def _spectra() -> np.ndarray:
    rng = np.random.default_rng(918)
    base = 1.0 + 0.1 * np.sin(np.linspace(0.0, 6.0, 81))
    return np.vstack(
        [
            base * scale + offset + rng.normal(0.0, 0.001, base.size)
            for scale, offset in [(0.8, -0.1), (1.0, 0.0), (1.2, 0.1), (1.1, -0.05)]
        ]
    )


def test_catalog_default_binds_verified_chemotools_and_native_methods():
    pipeline = PreprocessingPipeline(
        [
            PreprocessingStep(method="snv"),
            PreprocessingStep(method="sg_smooth", params={"window": 11, "order": 2}),
            PreprocessingStep(method="mean_center"),
        ]
    )
    providers = [item["provider"] for item in pipeline.provider_manifest()]
    assert providers == ["chemotools", "chemotools", "chemotools"]


def test_parameter_dependent_normalize_provider_is_bound_at_construction():
    l1 = PreprocessingPipeline(
        [PreprocessingStep(method="normalize", params={"norm": "l1"})]
    )
    l2 = PreprocessingPipeline(
        [PreprocessingStep(method="normalize", params={"norm": "l2"})]
    )
    maximum = PreprocessingPipeline(
        [PreprocessingStep(method="normalize", params={"norm": "max"})]
    )

    assert l1.provider_manifest()[0]["provider"] == "chemotools"
    assert l2.provider_manifest()[0]["provider"] == "chemotools"
    assert maximum.provider_manifest()[0]["provider"] == "native"


def test_norris_and_detrend_bind_chemotools_for_new_pipelines():
    pipeline = PreprocessingPipeline(
        [
            PreprocessingStep(method="detrend"),
            PreprocessingStep(
                method="norris_derivative1",
                params={"gap": 3, "segment": 5},
            ),
        ]
    )
    assert [item["provider"] for item in pipeline.provider_manifest()] == [
        "chemotools",
        "chemotools",
    ]


def test_rnv_is_a_distinct_chemotools_method_from_native_robust_snv():
    rnv_pipeline = PreprocessingPipeline(
        [PreprocessingStep(method="rnv", params={"percentile": 30.0})]
    )
    robust_snv_pipeline = PreprocessingPipeline(
        [PreprocessingStep(method="robust_snv")]
    )

    assert rnv_pipeline.provider_manifest()[0]["provider"] == "chemotools"
    assert robust_snv_pipeline.provider_manifest()[0]["provider"] == "native"
    assert not np.allclose(
        rnv_pipeline.apply(_spectra()),
        robust_snv_pipeline.apply(_spectra()),
    )


def test_new_chemotools_only_methods_bind_without_native_reimplementations():
    for method, params in (
        ("arpls", {"lambda_": 1e4, "ratio": 0.02, "max_iters": 30}),
        ("rubberband", {}),
        ("median_filter", {"window_length": 5}),
        ("whittaker_smooth", {"lambda_": 1e4}),
    ):
        pipeline = PreprocessingPipeline(
            [PreprocessingStep(method=method, params=params)]
        )
        manifest = pipeline.provider_manifest()[0]
        assert manifest["provider"] == "chemotools"
        assert pipeline.apply(_spectra()).shape == _spectra().shape


def test_native_policy_overrides_catalog_preference():
    pipeline = PreprocessingPipeline(
        [PreprocessingStep(method="snv"), PreprocessingStep(method="asls")],
        provider_policy="native",
    )
    assert [item["provider"] for item in pipeline.provider_manifest()] == [
        "native",
        "native",
    ]


def test_environment_can_hold_new_training_on_native(monkeypatch):
    monkeypatch.setenv("NIR_PREPROCESSING_PROVIDER_POLICY", "native")
    pipeline = PreprocessingPipeline([PreprocessingStep(method="snv")])
    assert pipeline.provider_manifest()[0]["provider"] == "native"


def test_unknown_provider_policy_fails_closed():
    with pytest.raises(ValueError, match="provider_policy"):
        PreprocessingPipeline(
            [PreprocessingStep(method="snv")],
            provider_policy="automatic-fallback",
        )


def test_provider_binding_survives_serialization(tmp_path):
    pipeline = PreprocessingPipeline(
        [PreprocessingStep(method="snv"), PreprocessingStep(method="mean_center")]
    ).fit(_spectra())
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    loaded = joblib.load(path)
    assert loaded.provider_manifest() == pipeline.provider_manifest()
    np.testing.assert_allclose(
        loaded.transform(_spectra()), pipeline.transform(_spectra())
    )


def test_legacy_pipeline_without_binding_is_pinned_to_native():
    pipeline = PreprocessingPipeline([PreprocessingStep(method="snv")])
    del pipeline._provider_bindings
    assert pipeline.provider_manifest()[0]["provider"] == "native"
    np.testing.assert_allclose(
        pipeline.apply(_spectra()),
        PreprocessingPipeline(
            [PreprocessingStep(method="snv")], provider_policy="native"
        ).apply(_spectra()),
    )


def test_stateful_chemotools_msc_matches_native_policy():
    X = _spectra()
    train, predict = X[:3], X[3:]
    chemotools_pipeline = PreprocessingPipeline([PreprocessingStep(method="msc")]).fit(
        train
    )
    native_pipeline = PreprocessingPipeline(
        [PreprocessingStep(method="msc")], provider_policy="native"
    ).fit(train)
    assert chemotools_pipeline.provider_manifest()[0]["provider"] == "chemotools"
    np.testing.assert_allclose(
        chemotools_pipeline.transform(predict),
        native_pipeline.transform(predict),
        rtol=1e-11,
        atol=1e-12,
    )


@pytest.mark.parametrize("method", ["mean_center", "autoscale"])
def test_stateful_chemotools_scalers_fit_on_training_only(method):
    from chemotools.scale import ParetoScaler

    X = _spectra()
    train, predict = X[:3], X[3:]
    pipeline = PreprocessingPipeline([PreprocessingStep(method=method)]).fit(train)
    expected_transformer = ParetoScaler(
        p=0.0 if method == "mean_center" else 1.0,
        with_mean=True,
        copy=True,
    ).fit(train)

    assert pipeline.provider_manifest()[0]["provider"] == "chemotools"
    np.testing.assert_allclose(
        pipeline.transform(predict),
        expected_transformer.transform(predict),
        rtol=0.0,
        atol=1e-12,
    )
    assert not np.allclose(
        pipeline.transform(predict),
        PreprocessingPipeline([PreprocessingStep(method=method)]).apply(predict),
    )


def test_stateful_chemotools_emsc_matches_direct_training_fit():
    from chemotools.scatter import ExtendedMultiplicativeScatterCorrection

    X = _spectra()
    train, predict = X[:3], X[3:]
    wv = np.linspace(1000.0, 1800.0, X.shape[1])
    pipeline = PreprocessingPipeline(
        [PreprocessingStep(method="emsc", params={"method": "median"})]
    ).fit(train, wv)
    expected = ExtendedMultiplicativeScatterCorrection(
        method="median",
        order=2,
    ).fit(train)

    assert pipeline.provider_manifest()[0]["provider"] == "chemotools"
    np.testing.assert_allclose(
        pipeline.transform(predict, wv),
        expected.transform(predict),
        rtol=0.0,
        atol=1e-12,
    )


def test_stateful_chemotools_airpls_matches_direct_training_fit():
    from chemotools.baseline import AirPls

    X = _spectra()
    train, predict = X[:3], X[3:]
    pipeline = PreprocessingPipeline(
        [
            PreprocessingStep(
                method="airpls",
                params={"lambda_": 1e5, "max_iters": 30},
            )
        ]
    ).fit(train)
    expected = AirPls(
        lam=1e5,
        nr_iterations=30,
        solver_type="banded",
        max_iter_after_warmstart=20,
        n_jobs=1,
    ).fit(train)

    assert pipeline.provider_manifest()[0]["provider"] == "chemotools"
    np.testing.assert_allclose(
        pipeline.transform(predict),
        expected.transform(predict),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("method", ["emsc", "airpls", "arpls"])
def test_stateful_chemotools_provider_state_survives_serialization(tmp_path, method):
    X = _spectra()
    train, predict = X[:3], X[3:]
    params = {"lambda_": 1e5, "max_iters": 30} if method in {"airpls", "arpls"} else {}
    pipeline = PreprocessingPipeline(
        [PreprocessingStep(method=method, params=params)]
    ).fit(train)
    expected = pipeline.transform(predict)
    path = tmp_path / f"{method}.joblib"
    joblib.dump(pipeline, path)
    loaded = joblib.load(path)

    assert loaded.provider_manifest() == pipeline.provider_manifest()
    np.testing.assert_allclose(loaded.transform(predict), expected)


def test_unfitted_copy_preserves_explicit_provider_policy():
    pipeline = PreprocessingPipeline(
        [PreprocessingStep(method="snv")],
        provider_policy="native",
    ).fit(_spectra())

    clone = pipeline.unfitted_copy()

    assert clone.fitted() is False
    assert clone.provider_manifest()[0]["provider"] == "native"


def test_unfitted_copy_pins_legacy_pipeline_to_native():
    pipeline = PreprocessingPipeline([PreprocessingStep(method="snv")])
    del pipeline._provider_bindings

    clone = pipeline.unfitted_copy()

    assert clone.provider_manifest()[0]["provider"] == "native"
