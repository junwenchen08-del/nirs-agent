"""Dependency contract for the pinned Chemotools preprocessing backend."""

from importlib.metadata import version


def test_pinned_chemotools_version_and_core_api_are_available():
    """The locked release must expose the classes planned by our adapters."""
    from chemotools.baseline import AirPls, ArPls, AsLs
    from chemotools.derivative import NorrisWilliams, SavitzkyGolay
    from chemotools.scatter import (
        ExtendedMultiplicativeScatterCorrection,
        MultiplicativeScatterCorrection,
        RobustNormalVariate,
        StandardNormalVariate,
    )
    from chemotools.smooth import MedianFilter, SavitzkyGolayFilter, WhittakerSmooth

    assert version("chemotools") == "0.4.4"
    assert all(
        item is not None
        for item in (
            AirPls,
            ArPls,
            AsLs,
            NorrisWilliams,
            SavitzkyGolay,
            ExtendedMultiplicativeScatterCorrection,
            MultiplicativeScatterCorrection,
            RobustNormalVariate,
            StandardNormalVariate,
            MedianFilter,
            SavitzkyGolayFilter,
            WhittakerSmooth,
        )
    )
