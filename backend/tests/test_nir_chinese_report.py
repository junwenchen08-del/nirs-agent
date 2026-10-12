"""Chinese presentation must use recorded evidence and preserve scientific scopes."""

import hashlib
import json
import zipfile

import numpy as np
import pytest
from test_nir_delivery_export import final_runtime, invoke_final


def test_autonomous_delivery_contains_offline_chinese_figures(tmp_path, monkeypatch):
    service, result, stop, output, runtime = final_runtime(tmp_path, monkeypatch)
    load = np.load

    def guarded_load(path, *args, **kwargs):
        # The first final evaluation is legal; the report must not reopen Test.
        if "holdout" in str(path):
            assert not getattr(guarded_load, "test_seen", False), "Report reopened Test"
            guarded_load.test_seen = True
        return load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_load)
    final = invoke_final(runtime)
    root = output / final["report"].removeprefix("/mnt/user-data/outputs/")
    report = root.read_text(encoding="utf-8")
    for phrase in ("原始与预处理后光谱", "验证误差趋势", "最终测试预测", "最终测试残差", "实际拟合次数", "岭回归", "主动停止", "折均值", "折外预测"):
        assert phrase in report
    images = list(root.parent.glob("*.png"))
    assert len(images) >= 4
    html = root.with_suffix(".html").read_text(encoding="utf-8")
    assert html.count("data:image/png;base64,") == len(images)
    with zipfile.ZipFile(root.parent / "delivery.zip") as archive:
        manifest = json.loads(archive.read("delivery_manifest.json"))
        for image in images:
            assert image.name in archive.namelist()
            assert archive.read(image.name) == image.read_bytes()
            assert any(item["path"] == image.name for item in manifest["files"])
    with service.search.process._connection() as db:
        session = service.search._session(db, final["session_id"])
    assert session["freeze"]["experiment_id"] == result["experiment_id"]
    assert all(any(name.endswith(image.name) for name in session["export_integrity"]) for image in images)
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("Read-only replay loaded an array"))
    assert invoke_final(runtime) == final
    assert guarded_load.test_seen is True
    assert (output / "report.md").read_text() == "historical report"


def test_figures_need_valid_persisted_cal_and_final_test_evidence(tmp_path, monkeypatch):
    from deerflow.community.nir.report_figures import build_report_figures

    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("Figure read a dataset"))
    plots = build_report_figures(tmp_path, {"decisions": [], "retained_experiment_id": "exp"}, cal_spectra={"scope": "final_test"}, final_chart={"scope": "validation"})
    assert plots == []
    assert not list(tmp_path.glob("*.png"))


def test_figures_keep_original_order_and_do_not_mutate_evidence(tmp_path):
    from deerflow.community.nir.report_figures import build_report_figures

    cal = {"scope": "calibration", "evidence_status": "available", "experiment_id": "exp", "source": "actual_cal_fitted_preprocessing_state"}
    for name in ("before", "after"):
        cal[name] = {"scope": "calibration_" + name, "x": [1200, 1190, 1180], "x_label": "wavelength", "series": [{"id": "sample", "values": [1, 2, 3]}], "n_total": 30, "n_returned": 1, "axis_total": 3, "axis_returned": 3}
    source = json.dumps(cal, sort_keys=True)
    figures = build_report_figures(tmp_path, {"decisions": [], "retained_experiment_id": "exp"}, cal_spectra=cal, final_chart={})
    assert len(figures) == 1
    assert "1/30" in figures[0]["explanation"]
    assert "单位未记录" in figures[0]["explanation"]
    assert json.dumps(cal, sort_keys=True) == source
    assert (tmp_path / figures[0]["filename"]).read_bytes().startswith(b"\x89PNG")
    assert hashlib.sha256((tmp_path / figures[0]["filename"]).read_bytes()).hexdigest()


def test_malformed_curve_pairs_fail_closed(tmp_path):
    from deerflow.community.nir.report_figures import build_report_figures

    with pytest.raises(ValueError, match="最终测试"):
        build_report_figures(tmp_path, {"decisions": []}, final_chart={"scope": "final_test", "evidence_status": "available", "observed": [1, 2], "predicted": [1], "residual": [0, 0]}, cal_spectra={})
