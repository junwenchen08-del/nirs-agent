"""Bounded modeling choice snapshots for the process explanation panel."""

from deerflow.community.nir.decision_facts import model_result_facts, selection_decision_facts


def test_selection_facts_keep_candidate_evidence_without_paths_or_arrays():
    facts = selection_decision_facts(
        {
            "preprocessing_selection": {
                "selected_candidate_id": "snv",
                "reason_code": "lowest_cv",
                "candidates": [{"candidate_id": "snv", "steps": ["snv"], "cv_rmse": 1.2, "path": "/mnt/private"}] * 100,
            },
            "wavelength_selection": {"method": "cars", "selected_indices": list(range(10000)), "n_selected": 42},
            "model_candidates": [{"method": "pls", "RMSE_tuning": 1.4, "raw_spectra": list(range(10000))}],
            "model_selection_decision": {"selected_method": "pls", "reason": "/mnt/private/result"},
            "model_path": "/mnt/private/model.pkl",
        }
    )
    assert len(facts["preprocessing_selection"]["candidates"]) == 8
    assert facts["preprocessing_selection"]["candidates"][0]["steps"] == ["snv"]
    assert facts["wavelength_selection"] == {"method": "cars", "n_selected": 42}
    assert facts["model_selection_decision"] == {"selected_method": "pls"}
    assert "/mnt/" not in str(facts)
    assert "raw_spectra" not in str(facts)


def test_adoption_improvement_is_kept_without_other_nested_payloads():
    facts = selection_decision_facts(
        {
            "model_selection_decision": {
                "selected_method": "pls",
                "adoption": {
                    "reason_code": "alternative_improvement_below_threshold",
                    "relative_RMSE_improvement": 0.003,
                    "minimum_required_improvement": 0.01,
                    "raw_predictions": list(range(1000)),
                },
            }
        }
    )
    assert facts["model_selection_decision"]["adoption"] == {
        "reason_code": "alternative_improvement_below_threshold",
        "relative_RMSE_improvement": 0.003,
        "minimum_required_improvement": 0.01,
    }


def test_model_result_facts_bound_legacy_labels_and_discard_bulk_candidates():
    facts = model_result_facts(
        {
            "task_kind": "classification",
            "classes": [f"class-{index}" for index in range(100)],
            "class_distribution": {f"class-{index}": index for index in range(100)},
            "candidate_results": [{"raw_spectra": list(range(1000))}],
            "method": "pls_da",
        }
    )
    assert len(facts["classes"]) == 20
    assert len(facts["class_distribution"]) == 20
    assert "candidate_results" not in facts
