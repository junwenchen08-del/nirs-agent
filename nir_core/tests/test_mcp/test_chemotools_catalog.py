from __future__ import annotations

from nir_core.chemotools_mcp.catalog import (
    CHEMOTOOLS_PINNED_VERSION,
    build_capability_catalog,
    describe_capability,
)


def test_catalog_covers_all_public_preprocessing_classes():
    catalog = build_capability_catalog()
    capabilities = {item["id"]: item for item in catalog["capabilities"]}

    expected = {
        # Baseline (10)
        "AirPls",
        "ArPls",
        "AsLs",
        "ConstantBaselineCorrection",
        "CubicSplineCorrection",
        "LinearCorrection",
        "NonNegative",
        "PolynomialCorrection",
        "RubberbandCorrection",
        "SubtractReference",
        # Derivative (2)
        "NorrisWilliams",
        "SavitzkyGolay",
        # Scale (5)
        "BandScaler",
        "MinMaxScaler",
        "NormScaler",
        "ParetoScaler",
        "PointScaler",
        # Scatter (4)
        "ExtendedMultiplicativeScatterCorrection",
        "MultiplicativeScatterCorrection",
        "RobustNormalVariate",
        "StandardNormalVariate",
        # Smooth (5)
        "MeanFilter",
        "MedianFilter",
        "ModifiedSincFilter",
        "SavitzkyGolayFilter",
        "WhittakerSmooth",
        # Projection (4)
        "DirectOrthogonalization",
        "ExternalParameterOrthogonalization",
        "OrthogonalPLS",
        "OrthogonalSignalCorrection",
    }
    preprocessing = {item["name"] for item in capabilities.values() if item["category"] in {"baseline", "derivative", "scale", "scatter", "smooth", "projection"} and item["kind"] != "constant"}

    assert catalog["provider_version"] == CHEMOTOOLS_PINNED_VERSION
    assert preprocessing == expected
    assert len(preprocessing) == 30


def test_catalog_covers_non_preprocessing_public_modules():
    catalog = build_capability_catalog()
    ids = {item["id"] for item in catalog["capabilities"]}

    assert "chemotools.adaptation.DirectStandardization" in ids
    assert "chemotools.augmentation.AddNoise" in ids
    assert "chemotools.feature_selection.VIPSelector" in ids
    assert "chemotools.regression.PLSRegression" in ids
    assert "chemotools.outliers.HotellingT2" in ids
    assert "chemotools.physics.IntensityConversion" in ids
    assert "chemotools.plotting.SpectraPlot" in ids
    assert "chemotools.inspector.PLSRegressionInspector" in ids
    assert "chemotools.datasets.load_coffee" in ids
    assert "chemotools.adaptation.functions.subtract_reference" in ids


def test_describe_uses_runtime_signature_and_parameter_docs():
    item = describe_capability("chemotools.smooth.SavitzkyGolayFilter")

    assert item["provider_version"] == CHEMOTOOLS_PINNED_VERSION
    assert item["kind"] == "transformer"
    assert "fit" in item["operations"]
    assert "transform" in item["operations"]
    assert "window_length" in item["parameters"]["properties"]
    assert item["parameters"]["properties"]["window_length"]["default"] == 3


def test_explicit_only_categories_do_not_masquerade_as_default_preprocessing():
    transfer = describe_capability("chemotools.adaptation.DirectStandardization")
    augmentation = describe_capability("chemotools.augmentation.AddNoise")
    snv = describe_capability("chemotools.scatter.StandardNormalVariate")

    assert transfer["explicit_only"] is True
    assert augmentation["explicit_only"] is True
    assert snv["explicit_only"] is False
