"""Real deterministic trainers must export process data, not frontend fixtures."""

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from deerflow.community.nir.process import ProcessStore
from deerflow.config.paths import Paths


@pytest.mark.parametrize(("entry", "knowledge_guided"), [("train", False), ("analyze", True), ("auto", False), ("partitioned", False), ("train", True), ("auto", True), ("partitioned", True)])
def test_actual_training_publishes_diagnostic_and_preprocessing_evidence(tmp_path, monkeypatch, entry, knowledge_guided):
    from deerflow.community.nir import modeling, single_target
    from deerflow.community.nir import process as process_module

    paths = Paths(tmp_path / "home")
    monkeypatch.setattr("deerflow.config.paths.get_paths", lambda: paths)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    rng = np.random.RandomState(137)
    latent = rng.normal(size=(96, 3))
    X = latent @ rng.normal(size=(3, 24)) + rng.normal(scale=0.01, size=(96, 24))
    y = 10 + latent @ np.array([2.0, -1.0, 0.5]) + rng.normal(scale=0.01, size=96)
    wv = np.linspace(1000, 2000, 24)
    source = tmp_path / "data.npz"
    np.savez(source, X=X, y=y, wv=wv)
    csv_source = tmp_path / "data.csv"
    import pandas as pd

    frame = pd.DataFrame(X, columns=[str(value) for value in wv])
    frame.insert(0, "reference", y)
    frame["partition"] = ["Cal"] * 65 + ["Tuning"] * 15 + ["Test"] * 16
    frame.to_csv(csv_source, index=False)

    def resolve(_runtime, virtual, *, read_only=True):
        if virtual.endswith("data.npz"):
            return str(source)
        if virtual.endswith("data.csv"):
            return str(csv_source)
        return str(outputs / virtual.split("/outputs/")[-1])

    monkeypatch.setattr(modeling, "_resolve", resolve)
    monkeypatch.setattr(single_target, "_resolve", resolve)
    monkeypatch.setattr("deerflow.community.nir._common._resolve", resolve)
    runtime = SimpleNamespace(state={"nir_workflow": {"project_id": "test-project", "unit": "%", "analyte": "protein"}}, context={"user_id": "alice"}, config={"configurable": {"thread_id": "thread"}})
    if entry == "train":
        result = single_target.nir_train_model_tool.func(
            runtime=runtime, input_path="/mnt/user-data/uploads/data.npz", pipeline_steps="auto" if knowledge_guided else '["mean_center"]', method="pls", max_components=5, cv_folds=3, tool_call_id="call-train"
        )
    elif entry == "analyze":
        result = modeling.nir_analyze_tool.func(runtime=runtime, data_path="/mnt/user-data/uploads/data.npz", auto_preprocess=True, method="pls", wavelength_selection="none", tool_call_id="call-analyze")
    else:
        kwargs = {
            "runtime": runtime,
            "file_path": "/mnt/user-data/uploads/data.csv",
            "y_col": 0,
            "x_cols": "1:25",
            "pipeline_steps": '["mean_center"]',
            "method": "pls",
            "max_components": 5,
            "compare_cars": False,
            "tool_call_id": f"call-{entry}",
        }
        if knowledge_guided:
            kwargs.pop("pipeline_steps")  # Exercise the actual new defaults.
        if entry == "auto":
            result = modeling.nir_train_auto_split_model_tool.func(**kwargs)
        else:
            result = modeling.nir_train_partitioned_model_tool.func(**kwargs, split_col="partition", train_label="Cal", tuning_label="Tuning", test_label="Test", validation_scope="independent_holdout_not_external")
    payload = json.loads(result)
    assert payload["status"] == "ok", payload
    if knowledge_guided:
        knowledge = payload["preprocessing_selection"]["method_knowledge"]
        assert knowledge["profile_scope"] == "calibration_only"
        assert knowledge["mode"] == "retrieval_guided"
        assert knowledge["sources"] and knowledge["sources"][0]["evidence_id"].startswith("method:")
    assert len(payload["deliverables"]) == 2
    html_path = outputs / payload["report_html"].split("/outputs/")[-1]
    html_text = html_path.read_text(encoding="utf-8")
    assert "最终留出集" in html_text and "调参集" in html_text
    assert "data:image/png;base64," in html_text
    if knowledge_guided:
        assert "数据诊断与算法知识依据" in html_text
        assert "https://chemotools.org/" in html_text
    assert ("reference" if entry == "auto" else "protein") in html_text and "校准集" in html_text
    with zipfile.ZipFile(outputs / payload["delivery_bundle"].split("/outputs/")[-1]) as archive:
        import hashlib

        records = json.loads(archive.read("delivery_manifest.json"))["files"]
        assert any(item["path"].endswith(".pkl.manifest.json") for item in records)
        assert any(item["path"].endswith(".json") and item["path"].startswith("evidence/") for item in records)
        for item in records:
            assert hashlib.sha256(archive.read(item["path"])).hexdigest() == item["sha256"]
    assert payload["process_evidence"]["evidence_status"] == "available", payload
    store = ProcessStore(paths.thread_dir("thread", user_id="alice") / "nir-process")
    run_id = payload["process_evidence"]["run_id"]
    summary = store.summary(run_id)
    assert summary["attempts"][0]["execution_status"] == "succeeded"
    steps = {step["step_key"]: store.step(run_id, step["step_execution_id"]) for step in summary["attempts"][0]["steps"]}
    schema = json.loads((Path(__file__).resolve().parents[2] / "contracts/nir-process-v1.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    validator.validate(summary)
    for step in steps.values():
        validator.validate(step)
        for ref in step["charts"]:
            validator.validate(store.chart(run_id, ref["chart_id"], ref["version"]))
    assert set(steps) == {"audit", "split", "preprocessing", "wavelength", "model", "validation"}
    before, after = [store.chart(run_id, ref["chart_id"], ref["version"]) for ref in steps["preprocessing"]["charts"]]
    assert [series["id"] for series in before["series"]] == [series["id"] for series in after["series"]]
    if not knowledge_guided:
        assert before["series"][0]["values"] != after["series"][0]["values"]
    prediction, residual = [store.chart(run_id, ref["chart_id"], ref["version"]) for ref in steps["validation"]["charts"][:2]]
    assert prediction["scope"] == "holdout"
    assert prediction["sample_ids"] == residual["sample_ids"]
    np.testing.assert_allclose(residual["y"], np.asarray(prediction["y"]) - np.asarray(prediction["x"]))
    rmse = float(np.sqrt(np.mean(np.asarray(residual["y"]) ** 2)))
    assert rmse == pytest.approx(steps["validation"]["metrics"]["test"]["RMSE"])
    assert steps["model"]["comparison_group_id"]
    if knowledge_guided:
        assert steps["preprocessing"]["facts"]["method_knowledge"]["profile_scope"] == "calibration_only"
        assert steps["preprocessing"]["facts"]["method_knowledge"]["sources"][0]["source_url"].startswith("https://chemotools.org/")
    assert "_model" not in process_module._json(steps)
    if entry == "analyze":
        second_result = json.loads(modeling.nir_analyze_tool.func(runtime=runtime, data_path="/mnt/user-data/uploads/data.npz", auto_preprocess=False, method="pls", wavelength_selection="none", tool_call_id="call-analyze-2"))
        assert second_result["process_evidence"]["evidence_status"] == "available"
        summary = store.summary(run_id)
        assert [item["number"] for item in summary["attempts"]] == [1, 2]
        models = [next(item for item in attempt["steps"] if item["step_key"] == "model") for attempt in summary["attempts"]]
        assert models[0]["comparison_group_id"] == models[1]["comparison_group_id"]
        assert store.chart(run_id, steps["validation"]["charts"][0]["chart_id"], steps["validation"]["charts"][0]["version"]) == prediction
