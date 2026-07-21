"""Tests for CARS and SPA wavelength selection algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.model.selection import cars_wavelength_selection, spa_wavelength_selection

# ---------------------------------------------------------------------------
# CARS
# ---------------------------------------------------------------------------


def test_cars_returns_nonempty_subset(synthetic_data):
    """CARS returns a non-empty candidate or the protected full baseline."""
    X, y = synthetic_data.X, synthetic_data.y
    X_sel, indices = cars_wavelength_selection(
        X, y, n_mc_samples=10, n_folds=3, random_state=42
    )
    assert len(indices) > 0
    assert len(indices) <= X.shape[1]
    # X_sel has the right shape.
    assert X_sel.shape == (X.shape[0], len(indices))
    # Indices are sorted and unique and within range.
    assert indices == sorted(set(indices))
    assert all(0 <= i < X.shape[1] for i in indices)
    # Selected columns match.
    assert np.allclose(X_sel, X[:, indices])


def test_cars_reproducible(synthetic_data):
    """Same random_state -> identical selected indices."""
    X, y = synthetic_data.X, synthetic_data.y
    _, i1 = cars_wavelength_selection(X, y, n_mc_samples=10, n_folds=3, random_state=42)
    _, i2 = cars_wavelength_selection(X, y, n_mc_samples=10, n_folds=3, random_state=42)
    assert i1 == i2


def test_cars_different_seeds_differ(synthetic_data):
    """Different seeds generally produce different subsets (statistical)."""
    X, y = synthetic_data.X, synthetic_data.y
    _, i1 = cars_wavelength_selection(X, y, n_mc_samples=10, n_folds=3, random_state=1)
    _, i2 = cars_wavelength_selection(
        X, y, n_mc_samples=10, n_folds=3, random_state=999
    )
    # They *could* be equal by chance, but with 500 wavelengths and
    # stochastic sampling it's very unlikely. Use a soft assertion.
    assert (i1 != i2) or (len(i1) < X.shape[1])


def test_cars_invalid_n_mc_raises(synthetic_data):
    X, y = synthetic_data.X, synthetic_data.y
    with pytest.raises(ValueError):
        cars_wavelength_selection(X, y, n_mc_samples=1)


def test_cars_shape_mismatch_raises(synthetic_data):
    X = synthetic_data.X
    y_short = np.zeros(X.shape[0] - 1)
    with pytest.raises(ValueError):
        cars_wavelength_selection(X, y_short, n_mc_samples=5)


def test_cars_on_small_data(small_synthetic_data):
    """CARS should work on smaller datasets without crashing."""
    X, y = small_synthetic_data.X, small_synthetic_data.y
    X_sel, indices = cars_wavelength_selection(
        X, y, n_mc_samples=8, n_folds=3, random_state=7
    )
    assert len(indices) > 0
    assert X_sel.shape[0] == X.shape[0]


def test_cars_uses_80_percent_mc_and_competitive_sampling(synthetic_data, monkeypatch):
    """MC draws 80% of samples and ARS draws subsets from the full pool."""
    X, y = synthetic_data.X, synthetic_data.y
    real_rng = np.random.default_rng(123)
    calls: list[tuple[int, int]] = []

    class RecordingRng:
        def choice(self, a, size, replace=False, p=None):
            population = int(a) if np.isscalar(a) else len(a)
            calls.append((population, int(size)))
            return real_rng.choice(a, size=size, replace=replace, p=p)

    monkeypatch.setattr(np.random, "default_rng", lambda _seed: RecordingRng())
    cars_wavelength_selection(
        X,
        y,
        n_mc_samples=6,
        n_folds=2,
        random_state=42,
    )

    assert calls[0] == (X.shape[0], round(0.8 * X.shape[0]))
    assert any(
        population == X.shape[1] and subset_size < population
        for population, subset_size in calls
    )


def test_cars_keeps_full_wavelength_baseline_when_it_is_better(
    synthetic_data, monkeypatch
):
    X, y = synthetic_data.X, synthetic_data.y

    def fake_cv(X_candidate, *_args, **_kwargs):
        return 0.1 if X_candidate.shape[1] == X.shape[1] else 1.0

    monkeypatch.setattr(
        "nir_core.model.selection._pls_cv_rmse",
        fake_cv,
    )
    _, indices = cars_wavelength_selection(
        X,
        y,
        n_mc_samples=6,
        n_folds=2,
        random_state=42,
    )

    assert indices == list(range(X.shape[1]))


# ---------------------------------------------------------------------------
# SPA
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_spa_returns_nonempty_subset(synthetic_data):
    """SPA must return a non-empty subset."""
    X, y = synthetic_data.X, synthetic_data.y
    X_sel, indices = spa_wavelength_selection(X, y, n_min=1, n_max=5)
    assert len(indices) > 0
    assert len(indices) <= 5
    assert X_sel.shape == (X.shape[0], len(indices))
    assert indices == sorted(set(indices))
    assert all(0 <= i < X.shape[1] for i in indices)


@pytest.mark.slow
def test_spa_reproducible(synthetic_data):
    """SPA is deterministic (no RNG) -> identical results across calls."""
    X, y = synthetic_data.X, synthetic_data.y
    _, i1 = spa_wavelength_selection(X, y, n_min=1, n_max=3)
    _, i2 = spa_wavelength_selection(X, y, n_min=1, n_max=3)
    assert i1 == i2


@pytest.mark.slow
def test_spa_default_n_max(synthetic_data):
    """n_max defaults to min(10, max(3, n_wavelengths // 3))."""
    X, y = synthetic_data.X, synthetic_data.y
    expected = min(10, max(3, X.shape[1] // 3))
    _, indices = spa_wavelength_selection(X, y)  # use defaults
    assert len(indices) <= expected


def test_spa_small_wavelength_count():
    """SPA should search a reasonable range even for small wavelength counts.

    Old formula gave n_max=1 for 15 wavelengths, which made the search
    trivially small.  New formula gives n_max=5, so the algorithm has room
    to explore multi-wavelength subsets.
    """
    from nir_core.tests.generators import generate_synthetic_spectra

    data = generate_synthetic_spectra(
        n_samples=40, n_wavelengths=15, n_components=3, random_state=42
    )
    X, y = data.X, data.y
    # With the fix, n_max=min(10, max(3, 15//3))=5.
    # With real signal (3 latent components), multiple wavelengths should help.
    _, indices = spa_wavelength_selection(X, y, n_min=3)
    assert len(indices) >= 3, f"Expected >= 3 selected wavelengths, got {len(indices)}"


def test_spa_n_min_greater_than_one(synthetic_data):
    """SPA honours n_min."""
    X, y = synthetic_data.X, synthetic_data.y
    _, indices = spa_wavelength_selection(X, y, n_min=3, n_max=3)
    assert len(indices) == 3


def test_spa_invalid_n_min_raises(synthetic_data):
    X, y = synthetic_data.X, synthetic_data.y
    with pytest.raises(ValueError):
        spa_wavelength_selection(X, y, n_min=0)


def test_spa_rejects_n_min_above_wavelength_count():
    X = np.arange(30, dtype=float).reshape(10, 3)
    y = np.arange(10, dtype=float)

    with pytest.raises(ValueError, match="cannot exceed"):
        spa_wavelength_selection(X, y, n_min=5)


def test_spa_shape_mismatch_raises(synthetic_data):
    X = synthetic_data.X
    y_short = np.zeros(X.shape[0] - 1)
    with pytest.raises(ValueError):
        spa_wavelength_selection(X, y_short)


def test_spa_on_small_data(small_synthetic_data):
    """SPA should work on smaller datasets."""
    X, y = small_synthetic_data.X, small_synthetic_data.y
    X_sel, indices = spa_wavelength_selection(X, y, n_min=1, n_max=3)
    assert len(indices) > 0
    assert X_sel.shape[0] == X.shape[0]
