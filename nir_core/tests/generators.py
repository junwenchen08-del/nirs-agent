"""Synthetic spectral data generators for unit testing.

All generators produce SpectralData with known ground truth so tests can
verify that algorithms recover the underlying structure.
"""

from __future__ import annotations

import numpy as np

from nir_core.models import SpectralData


def _gaussian_peak(x: np.ndarray, center: float, width: float, height: float) -> np.ndarray:
    """A single Gaussian peak over wavelength axis x."""
    return height * np.exp(-0.5 * ((x - center) / width) ** 2)


def generate_synthetic_spectra(
    n_samples: int = 200,
    n_wavelengths: int = 500,
    n_components: int = 3,
    noise_level: float = 0.01,
    random_state: int = 42,
) -> SpectralData:
    """Generate synthetic NIR spectra with known latent components.

    Construction:
    - Wavelengths: 1000-2500 nm (linear).
    - Latent scores: drawn from a standard normal, shape (n_samples, n_components).
    - Spectra: linear combination of n_components Gaussian peaks + baseline drift
      + additive Gaussian noise.
    - Reference y: linear combination of latent scores + small noise.

    Validation targets:
    - PLS should recover n_components.
    - Preprocessing should quantitatively improve SNR.
    - Three-way split should produce unbiased estimates.

    Args:
        n_samples: Number of samples.
        n_wavelengths: Number of wavelength points.
        n_components: Number of latent components (Gaussian peaks).
        noise_level: Std of additive Gaussian noise on spectra.
        random_state: Seed for reproducibility.

    Returns:
        SpectralData with X, y, wv populated.
    """
    rng = np.random.default_rng(random_state)

    wv = np.linspace(1000.0, 2500.0, n_wavelengths)

    # Peak centers spread across the wavelength range; each component has its
    # own peak shape.
    centers = np.linspace(1300.0, 2200.0, n_components)
    widths = np.full(n_components, 60.0)
    heights = np.full(n_components, 1.0)

    # Latent scores (n_samples, n_components)
    scores = rng.standard_normal((n_samples, n_components))

    # Build pure-component spectra (n_components, n_wavelengths)
    pure = np.vstack(
        [_gaussian_peak(wv, c, w, h) for c, w, h in zip(centers, widths, heights)]
    )

    # Spectra = scores @ pure + baseline drift + noise
    baseline = np.linspace(0.0, 0.2, n_wavelengths)  # shared linear drift
    X = scores @ pure + baseline[None, :]  # broadcast baseline to all samples
    X += rng.normal(0.0, noise_level, size=X.shape)

    # Reference y = linear combination of latent scores (known coefficients)
    coef = np.linspace(1.0, 2.0, n_components)
    y = scores @ coef + rng.normal(0.0, 0.05, size=n_samples)

    sample_names = [f"sample_{i:04d}" for i in range(n_samples)]

    return SpectralData(
        X=X,
        y=y,
        wv=wv,
        sample_names=sample_names,
        source_file="<synthetic>",
        original_format="synthetic",
    )


def generate_corn_like_spectra(
    n_samples: int = 80, n_wavelengths: int = 700, random_state: int = 42
) -> SpectralData:
    """Generate Corn-dataset-like synthetic spectra.

    Mimics the structure of the Eigenvector Research Corn dataset with four
    constituents (moisture, oil, protein, starch). Only ``moisture`` is
    returned as y for simplicity; the rest are latent in the spectra.

    Args:
        n_samples: Number of samples (Corn has 80).
        n_wavelengths: Number of wavelength points (Corn has ~700).
        random_state: Seed.

    Returns:
        SpectralData with y = moisture content.
    """
    rng = np.random.default_rng(random_state)
    wv = np.linspace(1100.0, 2500.0, n_wavelengths)

    # Four latent constituents with distinct peak locations.
    n_comp = 4
    centers = [1450.0, 1730.0, 1940.0, 2100.0]  # water, oil, protein, starch
    widths = [50.0, 60.0, 70.0, 55.0]
    pure = np.vstack(
        [_gaussian_peak(wv, c, w, 1.0) for c, w in zip(centers, widths)]
    )

    # Constituent concentrations (n_samples, 4), realistic-ish ranges.
    moisture = rng.uniform(8.0, 13.0, n_samples)
    oil = rng.uniform(3.0, 5.5, n_samples)
    protein = rng.uniform(7.5, 11.0, n_samples)
    starch = rng.uniform(60.0, 70.0, n_samples)
    conc = np.column_stack([moisture, oil, protein, starch])

    # Normalize each pure component so scaling is reasonable.
    scale = np.array([1.0, 1.0, 1.0, 0.05])
    X = conc * scale[None, :] @ pure  # (n_samples, n_wavelengths)
    X += rng.normal(0.0, 0.01, size=X.shape)

    sample_names = [f"corn_{i:03d}" for i in range(n_samples)]
    return SpectralData(
        X=X,
        y=moisture,
        wv=wv,
        sample_names=sample_names,
        source_file="<synthetic_corn>",
        original_format="synthetic",
    )


def generate_noisy_spectra(
    n_samples: int = 100,
    n_wavelengths: int = 300,
    noise_level: float = 0.1,
    random_state: int = 0,
) -> SpectralData:
    """Generate spectra with controllable noise level (for SNR tests)."""
    return generate_synthetic_spectra(
        n_samples=n_samples,
        n_wavelengths=n_wavelengths,
        noise_level=noise_level,
        random_state=random_state,
    )
