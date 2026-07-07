"""NIR agent configuration, managed inside the package to avoid
modifying DeerFlow's AppConfig (which is a Pydantic model that rejects
undefined fields).

Configuration can be overridden by a JSON file on disk; otherwise
defaults are used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class QualityThresholds:
    """Quality-gate thresholds, tiered by application domain."""

    min_r2: float = 0.80
    min_rpd: float = 3.0
    max_retries: int = 3

    # Per-domain thresholds (override the defaults above).
    domain_thresholds: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "food_protein": {"min_r2": 0.85, "min_rpd": 3.5},
            "food_moisture": {"min_r2": 0.90, "min_rpd": 4.0},
            "soil": {"min_r2": 0.70, "min_rpd": 2.0},
            "feed": {"min_r2": 0.75, "min_rpd": 2.5},
            "pharma": {"min_r2": 0.85, "min_rpd": 3.0},
            "default": {"min_r2": 0.80, "min_rpd": 3.0},
        }
    )

    def get_thresholds(
        self, domain: str = "default", n_samples: int | None = None
    ) -> dict[str, Any]:
        """Return effective thresholds.

        For small samples (<100), thresholds are relaxed by R2-0.10 / RPD-0.5
        (floored at 0.60 / 1.5) to avoid over-retrying on small datasets.
        """
        base = self.domain_thresholds.get(domain, self.domain_thresholds["default"])
        r2 = float(base["min_r2"])
        rpd = float(base["min_rpd"])
        if n_samples is not None and n_samples < 100:
            r2 = max(0.60, r2 - 0.10)
            rpd = max(1.5, rpd - 0.5)
        return {
            "min_r2": r2,
            "min_rpd": rpd,
            "max_retries": self.max_retries,
            "domain": domain,
            "n_samples": n_samples,
        }


@dataclass
class NirConfig:
    """Global NIR agent configuration."""

    quality: QualityThresholds = field(default_factory=QualityThresholds)

    defaults: dict[str, Any] = field(
        default_factory=lambda: {
            "test_ratio": 0.20,
            "val_ratio": 0.10,
            "cv_folds": 10,
            "max_components": 20,
        }
    )

    candidate_pipelines: list[list[str]] = field(
        default_factory=lambda: [
            ["snv", "sg_smooth", "mean_center"],
            ["msc", "derivative1", "autoscale"],
            ["airpls", "sg_smooth", "snv", "mean_center"],
        ]
    )

    registry_path: str = "artifacts/registry.json"

    drift: dict[str, Any] = field(
        default_factory=lambda: {
            "mahalanobis_threshold": 3.0,
            "enabled": True,
        }
    )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "NirConfig":
        """Load config from a JSON file; fall back to defaults if missing."""
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        quality = QualityThresholds(**data.get("quality", {}))
        return cls(
            quality=quality,
            defaults=data.get("defaults", {}),
            candidate_pipelines=data.get("candidate_pipelines", []),
            registry_path=data.get("registry_path", "artifacts/registry.json"),
            drift=data.get("drift", {}),
        )

    def to_dict(self) -> dict:
        return {
            "quality": {
                "min_r2": self.quality.min_r2,
                "min_rpd": self.quality.min_rpd,
                "max_retries": self.quality.max_retries,
                "domain_thresholds": self.quality.domain_thresholds,
            },
            "defaults": self.defaults,
            "candidate_pipelines": self.candidate_pipelines,
            "registry_path": self.registry_path,
            "drift": self.drift,
        }


_config: NirConfig | None = None


def get_nir_config() -> NirConfig:
    """Return the global NIR config singleton (defaults if unset)."""
    global _config
    if _config is None:
        _config = NirConfig()
    return _config


def set_nir_config(config: NirConfig) -> None:
    """Override the global NIR config (e.g. from a loaded JSON file)."""
    global _config
    _config = config
