"""Pytest fixtures shared across the nir_core test suite."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.models import SpectralData
from nir_core.tests.generators import (
    generate_corn_like_spectra,
    generate_synthetic_spectra,
)


@pytest.fixture(scope="session")
def synthetic_data() -> SpectralData:
    """A standard synthetic dataset (200 samples x 500 wavelengths, 3 latent)."""
    return generate_synthetic_spectra(
        n_samples=200, n_wavelengths=500, n_components=3, random_state=42
    )


@pytest.fixture(scope="session")
def small_synthetic_data() -> SpectralData:
    """A smaller dataset for fast tests (60 samples x 200 wavelengths)."""
    return generate_synthetic_spectra(
        n_samples=60, n_wavelengths=200, n_components=2, random_state=7
    )


@pytest.fixture(scope="session")
def corn_like_data() -> SpectralData:
    """A Corn-like synthetic dataset (80 samples, y = moisture)."""
    return generate_corn_like_spectra(n_samples=80, n_wavelengths=700, random_state=42)


@pytest.fixture
def rng() -> np.random.Generator:
    """A fresh seeded Generator per test."""
    return np.random.default_rng(123)


@pytest.fixture
def linear_data() -> tuple[np.ndarray, np.ndarray]:
    """Perfectly linear y = 2x + 1 data for metric sanity checks."""
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1
    return x, y
