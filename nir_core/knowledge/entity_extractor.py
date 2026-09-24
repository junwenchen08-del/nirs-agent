"""NIR-domain entity extractor.

Extracts method / model / metric / dataset mentions from Markdown text.
The extracted entities are stored in ChromaDB chunk metadata, enabling
metadata-filtered retrieval today and Neo4j graph edges in the future
without re-parsing source documents.

Matching is intentionally conservative: whole-word, case-insensitive,
using curated NIR-specific dictionaries plus an alias map that normalises
common paper-text variants (e.g. "SG smoothing" -> "sg_smooth") to the
canonical names used by ``nir_core``. This keeps precision high and avoids
the complexity / weight of an NER model.
"""

from __future__ import annotations

import re

# NIR preprocessing methods (must align with nir_core PRESTEP_METHODS)
_PREPROCESSING_METHODS = {
    "snv",
    "robust_snv",
    "msc",
    "emsc",
    "despike",
    "sg_smooth",
    "savitzky-golay",
    "derivative",
    "derivative1",
    "derivative2",
    "norris_derivative1",
    "norris_derivative2",
    "airpls",
    "asls",
    "detrend",
    "mean_center",
    "autoscale",
    "normalize",
    "snv_detrend",
    "osc",
    "ems",
    "baseline_correction",
}

# Modeling methods
_MODELING_METHODS = {
    "pls",
    "pcr",
    "svr",
    "svr_rbf",
    "svr_linear",
    "svr_poly",
    "ridge",
    "lasso",
    "elasticnet",
    "random_forest",
    "xgboost",
    "ann",
    "cnn",
    "lstm",
    "plsr",
    "mlp",
    "lightgbm",
    "catboost",
}

# Variable selection / wavelength selection methods
_VARIABLE_SELECTION = {
    "spa",
    "cars",
    "uve",
    "vip",
    "mcuves",
    "irf",
    "boruta",
    "genetic_algorithm",
    "ga",
    "sipl",
}

# Evaluation metrics
_METRICS = {
    "r2",
    "rmse",
    "rmsecv",
    "rmsep",
    "rpd",
    "rpdq",
    "bias",
    "slope",
    "mae",
    "mse",
    "aicc",
    "sec",
    "sep",
}

# Sample / dataset domains
_DATASETS = {
    "soil",
    "corn",
    "wheat",
    "rice",
    "barley",
    "oat",
    "meat",
    "milk",
    "fruit",
    "coffee",
    "tea",
    "pharmaceutical",
    "tablet",
    "powder",
    "forage",
    "silage",
}

_ENTITIES: dict[str, set[str]] = {
    "methods": _PREPROCESSING_METHODS,
    "models": _MODELING_METHODS,
    "variable_selection": _VARIABLE_SELECTION,
    "metrics": _METRICS,
    "datasets": _DATASETS,
}

# Alias map: textual variant -> canonical entity name (per category).
# Paper prose often uses natural-language forms like "SG smoothing" instead
# of the code identifier "sg_smooth". Without aliasing these would be missed.
# Chinese aliases are included because many NIR theses are written in Chinese,
# and the English acronym (PLS, SNV, ...) may appear only in references.
_ALIASES: dict[str, dict[str, str]] = {
    "methods": {
        "sg smoothing": "sg_smooth",
        "savitzky golay": "savitzky-golay",
        "savitzky–golay": "savitzky-golay",  # en-dash variant
        "mean centering": "mean_center",
        "auto scale": "autoscale",
        "robust snv": "robust_snv",
        "extended multiplicative scatter correction": "emsc",
        "emsc": "emsc",
        "despiking": "despike",
        "spike removal": "despike",
        "norris williams first derivative": "norris_derivative1",
        "norris-williams first derivative": "norris_derivative1",
        "norris williams second derivative": "norris_derivative2",
        "norris-williams second derivative": "norris_derivative2",
        "first derivative": "derivative1",
        "second derivative": "derivative2",
        "air pls": "airpls",
        "as ls": "asls",
        "baseline correction": "baseline_correction",
        # Chinese terms (common in NIR theses)
        "标准正态变量变换": "snv",
        "标准正态变量": "snv",
        "稳健标准正态变量": "robust_snv",
        "稳健snv": "robust_snv",
        "多元散射校正": "msc",
        "扩展多元散射校正": "emsc",
        "去尖峰": "despike",
        "尖峰去除": "despike",
        "平滑去噪": "sg_smooth",
        "一阶导数": "derivative1",
        "二阶导数": "derivative2",
        "均值中心化": "mean_center",
        "自动缩放": "autoscale",
        "基线校正": "baseline_correction",
    },
    "models": {
        "partial least squares": "pls",
        "support vector regression": "svr",
        "random forest": "random_forest",
        "neural network": "ann",
        "artificial neural network": "ann",
        # Chinese terms
        "偏最小二乘": "pls",
        "支持向量回归": "svr",
        "随机森林": "random_forest",
        "神经网络": "ann",
        "人工神经网络": "ann",
        "岭回归": "ridge",
        "套索回归": "lasso",
    },
    "datasets": {
        "土壤": "soil",
        "玉米": "corn",
        "小麦": "wheat",
        "水稻": "rice",
        "大麦": "barley",
        "肉类": "meat",
        "牛奶": "milk",
        "水果": "fruit",
        "咖啡": "coffee",
        "茶叶": "tea",
        "药物": "pharmaceutical",
        "药片": "tablet",
        "饲料": "forage",
    },
}


def extract_entities(markdown: str) -> dict[str, list[str]]:
    """Extract NIR-domain entities from text.

    Args:
        markdown: Document text (Markdown or plain).

    Returns:
        Dict with keys ``methods``, ``models``, ``variable_selection``,
        ``metrics``, ``datasets``, each mapping to a sorted list of matched
        entity names (canonical form).
    """
    text_lower = markdown.lower()

    # CJK Unicode range (for detecting whether \b word boundaries apply).
    def _is_cjk(value: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in value)

    def _find(terms: set[str], aliases: dict[str, str]) -> list[str]:
        found: set[str] = set()
        for term in terms:
            if _is_cjk(term):
                # \b doesn't work for CJK (non-word chars in Python regex);
                # use plain substring match.
                if term in text_lower:
                    found.add(term)
            else:
                pattern = r"\b" + re.escape(term) + r"\b"
                if re.search(pattern, text_lower):
                    found.add(term)
        # Check aliases — map textual variant to canonical name.
        for variant, canonical in aliases.items():
            if _is_cjk(variant):
                if variant in text_lower and canonical in terms:
                    found.add(canonical)
            else:
                pattern = r"\b" + re.escape(variant) + r"\b"
                if re.search(pattern, text_lower) and canonical in terms:
                    found.add(canonical)
        return sorted(found)

    return {
        category: _find(terms, _ALIASES.get(category, {}))
        for category, terms in _ENTITIES.items()
    }


def merge_entities(*entity_dicts: dict[str, list[str]]) -> dict[str, list[str]]:
    """Merge multiple entity dicts (deduplicated, sorted).

    Useful when extracting entities from multiple chunks of the same doc.
    """
    merged: dict[str, set[str]] = {cat: set() for cat in _ENTITIES}
    for ed in entity_dicts:
        for cat, terms in ed.items():
            if cat in merged:
                merged[cat].update(terms)
    return {cat: sorted(terms) for cat, terms in merged.items()}
