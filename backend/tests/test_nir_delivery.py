"""Reports and delivery packages must preserve actual scientific evidence."""

import hashlib
import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from deerflow.community.nir._report import _build_report


def test_report_distinguishes_tuning_holdout_and_missing_axis_unit():
    data = SimpleNamespace(X=np.zeros((12, 5)), y=np.arange(12), wv=np.arange(5))
    metrics = {
        "n_samples": 12,
        "method": "pls",
        "n_components": 2,
        "R2_val": 0.99,
        "RMSEP": 2.0,
        "RPD": 3.0,
        "val": {"R2": 0.99, "RMSE": 0.1},
        "test": {"R2": 0.75, "RMSE": 2.0, "RPD": 3.0, "bias": -0.2},
        "validation_scope": "independent_holdout_not_external",
        "wavelength_selection": {"method": "none", "n_selected": 5, "n_original": 5},
        "model_candidates": [{"method": "pls", "RMSE_tuning": 0.1}],
    }
    quality = {"grade": "good", "passed": True, "action": "proceed", "thresholds_used": {"min_r2": 0.8, "min_rpd": 3.0, "domain": "food"}}
    report = _build_report(data, metrics, quality, None, raw_spectra_b64="placeholder", predicted_vs_reference_b64="placeholder")
    assert "调参集" in report and "最终留出集" in report
    assert "0.7500" in report and "0.9900" in report
    assert "单位未记录" in report and " nm" not in report
    assert "![原始光谱](raw_spectra.png)" in report
    assert "不等同于生产批准" in report
    assert "data:image" not in report


def test_delivery_embeds_only_declared_figures_and_packages_external_metrics(tmp_path):
    from deerflow.community.nir.delivery import build_delivery

    root = tmp_path / "outputs"
    root.mkdir()
    report = root / "report.md"
    report.write_text("# Report\n\n![plot](raw_spectra.png)\n\n<script>alert(1)</script>\n", encoding="utf-8")
    plot = root / "raw_spectra.png"
    plot.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    model = root / "custom.pkl"
    model.write_bytes(b"model")
    manifest = root / "custom.pkl.manifest.json"
    manifest.write_text("{}")
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"test":{"RMSE":2}}')
    (root / "private-upload.csv").write_text("must not be packaged")
    result = build_delivery(report, model=model, metrics=metrics, figures=[plot])
    html = (root / "report.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in html
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert result["bundle"].name == "delivery.zip"
    with zipfile.ZipFile(result["bundle"]) as archive:
        names = archive.namelist()
        assert "private-upload.csv" not in names
        assert "model/custom.pkl" in names and "model/custom.pkl.manifest.json" in names
        assert "evidence/metrics.json" in names
        records = json.loads(archive.read("delivery_manifest.json"))["files"]
        for record in records:
            assert hashlib.sha256(archive.read(record["path"])).hexdigest() == record["sha256"]


def test_delivery_rejects_symlink_and_nonlocal_image(tmp_path):
    from deerflow.community.nir.delivery import build_delivery

    report = tmp_path / "report.md"
    report.write_text("![bad](../../private.png)\n![remote](https://example.com/image.png)", encoding="utf-8")
    result = build_delivery(report)
    html = result["html"].read_text(encoding="utf-8")
    assert "<img" not in html and "data:image" not in html
    source = tmp_path / "source.png"
    source.write_bytes(b"secret")
    link = tmp_path / "link.png"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symlink"):
        build_delivery(report, figures=[link])


def test_collection_combines_child_reports_and_preserves_partial_failures(tmp_path, monkeypatch):
    from deerflow.community.nir import modeling

    root = tmp_path / "collection"
    monkeypatch.setattr(modeling, "_resolve", lambda *args, **kwargs: str(tmp_path / "input.mat"))
    monkeypatch.setattr(modeling, "_resolve_writable_dir", lambda *args, **kwargs: str(root))
    monkeypatch.setattr(modeling, "_collection_subset_names", lambda *args: ["R562", "R568", "failed"])

    def analyze(**kwargs):
        name = kwargs["subset"]
        if name == "failed":
            return json.dumps({"status": "error", "error": "recorded failure"})
        child = root / name
        child.mkdir()
        (child / "report.md").write_text(f"# {name} detailed report\n![spectrum](raw_spectra.png)", encoding="utf-8")
        (child / "model.pkl").write_bytes(b"model")
        (child / "model.pkl.manifest.json").write_text("{}")
        (child / "metrics.json").write_text("{}")
        (child / "raw_spectra.png").write_bytes(b"PNG")
        virtual = kwargs["output_dir"]
        return json.dumps({"status": "ok", "R2_val": 0.9, "passed": True, "report": virtual + "/report.md", "model": virtual + "/model.pkl", "metrics": virtual + "/metrics.json", "plots": {"raw": virtual + "/raw_spectra.png"}})

    monkeypatch.setattr(modeling, "_analyze_collection_item", analyze)
    payload = json.loads(modeling.nir_analyze_collection_tool.func(runtime=SimpleNamespace(state={}), data_path="/mnt/user-data/uploads/input.mat"))
    assert payload["status"] == "ok" and payload["passed"] is False
    assert payload["artifacts"] == payload["deliverables"] and len(payload["artifacts"]) == 2
    report = (root / "collection_summary.html").read_text(encoding="utf-8")
    assert "R562 detailed report" in report and "R568 detailed report" in report
    assert "recorded failure" in report
    assert report.count("data:image/png;base64,") == 2
    with zipfile.ZipFile(root / "delivery.zip") as archive:
        assert "R562/model.pkl.manifest.json" in archive.namelist()
        assert "R568/report.md" in archive.namelist()


@pytest.mark.parametrize("kind", ["raw", "vip", "coefficients"])
def test_report_plot_keeps_axis_values_without_inventing_units(monkeypatch, kind):
    from nir_core.plotting import model_diag, spectra

    axis = np.array([5545.8, 5800, 6975.2])

    def inspect_figure(figure):
        import matplotlib.pyplot as plt

        label = figure.axes[0].get_xlabel()
        plt.close(figure)
        return label

    monkeypatch.setattr(spectra, "_fig_to_base64", inspect_figure)
    monkeypatch.setattr(model_diag, "_fig_to_base64", inspect_figure)
    if kind == "raw":
        result = spectra.plot_raw_spectra(SimpleNamespace(X=np.ones((3, 3)), wv=axis), wavelength_unit=None)
    elif kind == "vip":
        result = model_diag.plot_vip(np.ones(3), wv=axis, wavelength_unit=None)
    else:
        result = model_diag.plot_regression_coefficients(np.ones(3), wv=axis, wavelength_unit=None)
    assert result == "Spectral axis (unit not recorded)"
