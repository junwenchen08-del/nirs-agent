"""Tests for the NIR domain entity extractor."""

from __future__ import annotations

from nir_core.knowledge.entity_extractor import extract_entities, merge_entities


def test_extract_methods() -> None:
    """Preprocessing methods are detected."""
    text = "We applied SNV followed by SG smoothing and mean centering."
    entities = extract_entities(text)
    assert "snv" in entities["methods"]
    assert "sg_smooth" in entities["methods"]
    assert "mean_center" in entities["methods"]


def test_extract_models() -> None:
    """Modeling methods are detected."""
    text = "PLS and SVR were compared; random_forest also tried."
    entities = extract_entities(text)
    assert "pls" in entities["models"]
    assert "svr" in entities["models"]
    assert "random_forest" in entities["models"]


def test_extract_datasets() -> None:
    """Dataset domains are detected."""
    text = "Soil samples and corn kernels were analysed."
    entities = extract_entities(text)
    assert "soil" in entities["datasets"]
    assert "corn" in entities["datasets"]


def test_extract_metrics() -> None:
    """Metrics are detected."""
    text = "The model achieved R2=0.9, RMSEP=0.3, RPD=3.2."
    entities = extract_entities(text)
    assert "r2" in entities["metrics"]
    assert "rmsep" in entities["metrics"]
    assert "rpd" in entities["metrics"]


def test_extract_variable_selection() -> None:
    """Variable selection methods are detected."""
    text = "SPA and CARS were used for wavelength selection."
    entities = extract_entities(text)
    assert "spa" in entities["variable_selection"]
    assert "cars" in entities["variable_selection"]


def test_extract_no_entities() -> None:
    """Text without NIR entities returns empty lists."""
    text = "This is a generic text about machine learning and statistics."
    entities = extract_entities(text)
    for key in ("methods", "models", "variable_selection", "metrics", "datasets"):
        assert entities[key] == []


def test_extract_case_insensitive() -> None:
    """Entity matching is case-insensitive."""
    text = "We used SNV, Snv, and snv in different sentences."
    entities = extract_entities(text)
    assert "snv" in entities["methods"]


def test_extract_returns_sorted() -> None:
    """Entity lists are sorted."""
    text = "SVR and PLS and ANN were compared."
    entities = extract_entities(text)
    models = entities["models"]
    assert models == sorted(models)


def test_extract_whole_word_match() -> None:
    """Matching uses word boundaries (no partial matches)."""
    # "snv" should not match "snvXYZ" or "XYZsnv".
    text = "The snvXYZ method and XYZsnv are not real methods."
    entities = extract_entities(text)
    assert "snv" not in entities["methods"]


def test_merge_entities() -> None:
    """merge_entities deduplicates and sorts."""
    e1 = {"methods": ["snv", "msc"], "models": ["pls"]}
    e2 = {"methods": ["snv", "airpls"], "models": ["svr"]}
    merged = merge_entities(e1, e2)
    assert merged["methods"] == ["airpls", "msc", "snv"]
    assert merged["models"] == ["pls", "svr"]


def test_merge_entities_empty() -> None:
    """merge_entities handles empty dicts."""
    merged = merge_entities({}, {})
    assert merged["methods"] == []


# ---------------------------------------------------------------------------
# Chinese-language alias matching (common in NIR theses)
# ---------------------------------------------------------------------------


def test_extract_chinese_method_aliases() -> None:
    """Chinese preprocessing method names are detected via aliases."""
    text = "本文采用标准正态变量变换(SNV)和多元散射校正(MSC)进行散射校正。"
    entities = extract_entities(text)
    assert "snv" in entities["methods"]
    assert "msc" in entities["methods"]


def test_extract_chinese_model_aliases() -> None:
    """Chinese model names are detected via aliases."""
    text = "偏最小二乘(PLS)和支持向量回归(SVR)是常用的定量分析模型。"
    entities = extract_entities(text)
    assert "pls" in entities["models"]
    assert "svr" in entities["models"]


def test_extract_chinese_dataset_aliases() -> None:
    """Chinese dataset names are detected via aliases."""
    text = "实验使用了土壤和小麦样本。"
    entities = extract_entities(text)
    assert "soil" in entities["datasets"]
    assert "wheat" in entities["datasets"]


def test_extract_chinese_derivative_alias() -> None:
    """Chinese derivative terms are detected."""
    text = "一阶导数和二阶导数常用于消除基线漂移。"
    entities = extract_entities(text)
    assert "derivative1" in entities["methods"]
    assert "derivative2" in entities["methods"]


def test_extract_new_preprocessing_method_aliases() -> None:
    text = (
        "We compared robust SNV, extended multiplicative scatter correction, "
        "despiking, and a Norris-Williams first derivative."
    )
    entities = extract_entities(text)
    assert {"robust_snv", "emsc", "despike", "norris_derivative1"}.issubset(
        set(entities["methods"])
    )


def test_extract_new_chinese_preprocessing_aliases() -> None:
    entities = extract_entities("采用稳健SNV、扩展多元散射校正和去尖峰预处理。")
    assert {"robust_snv", "emsc", "despike"}.issubset(set(entities["methods"]))


def test_extract_chinese_neural_network_alias() -> None:
    """Chinese neural network term is detected as 'ann'."""
    text = "人工神经网络在光谱分析中应用广泛。"
    entities = extract_entities(text)
    assert "ann" in entities["models"]
