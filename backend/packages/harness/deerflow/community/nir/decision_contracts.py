"""Strict initial and feedback decisions; actions are selected by the LLM."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Text = Annotated[str, Field(min_length=1, max_length=2000)]
Identifier = Annotated[str, Field(min_length=1, max_length=128)]
Action = Literal["EXPLORE", "TUNE", "CHANGE_PREPROCESS", "SWITCH_MODEL", "STOP", "RETAIN"]


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class PreprocessingSpec(StrictContract):
    method: Identifier
    params: dict = Field(default_factory=dict)


class PipelineSpec(StrictContract):
    preprocessing: list[PreprocessingSpec] = Field(max_length=5)
    model_method: Literal["pls", "ridge", "svr", "et"]
    fixed_params: dict = Field(default_factory=dict)
    wavelength_selection: Literal["none"] = "none"


class SearchSpec(StrictContract):
    strategy: Literal["fixed", "grid"]
    search_space: dict[str, list] = Field(default_factory=dict)
    max_trials: int = Field(ge=1, le=40, description="Maximum Cartesian grid candidates (fixed has one candidate).")
    max_fits: int = Field(ge=1, le=200, description="All CV fits plus one full-Cal refit: n_candidates * frozen n_splits + 1. Fixed parameters still receive CV evaluation.")
    objective: Literal["train_cv_rmse"] = "train_cv_rmse"


class EvidenceClaim(StrictContract):
    claim: Text
    evidence_refs: list[Identifier] = Field(min_length=1, max_length=20)


class Alternative(StrictContract):
    id: Identifier
    action: Action = "EXPLORE"
    pipeline: PipelineSpec | None
    optimization: SearchSpec | None
    comparison: Text


class DecisionProposal(StrictContract):
    context_id: Identifier
    expected_revision: int = Field(ge=1)
    context_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    phase: Literal["initial", "feedback", "final_selection"]
    action: Action
    observations: list[EvidenceClaim] = Field(min_length=1, max_length=12)
    evidence_refs: list[Identifier] = Field(min_length=1, max_length=20)
    alternatives: list[Alternative] = Field(min_length=2, max_length=4)
    selected_alternative_id: Identifier
    rationale: Text
    expected_outcome: Text
    base_experiment_id: Identifier | None = None
    retained_experiment_id: Identifier | None = None
    uncertainties: list[Text] = Field(default_factory=list, max_length=6)
    not_selected_reasons: dict[str, Text] = Field(default_factory=dict)
    hypotheses: list[Text] = Field(default_factory=list, max_length=6)
    expected_fits: int | None = Field(default=None, ge=0, le=200)
    expected_improvement: "ImprovementGoal | None" = None
    difference: Text | None = None
    replication_reason: Text | None = None
    stop_evidence: "StopEvidence | None" = None


class UnverifiedDirection(StrictContract):
    direction: Text
    reason: Text


class StopEvidence(StrictContract):
    retained_advantages: list[Text] = Field(min_length=1, max_length=4)
    retained_limitations: list[Text] = Field(min_length=1, max_length=4)
    search_coverage: Text
    unverified_directions: list[UnverifiedDirection] = Field(min_length=1, max_length=6)


class ImprovementGoal(StrictContract):
    metric: Literal["val_rmse", "val_r2"]
    comparison: Literal["previous", "retained"]
    minimum_change: float = Field(ge=0)


class DecisionContext(StrictContract):
    """Server-produced snapshot; feedback history is read from private records."""

    schema_version: Literal[1, 2, 3]
    context_id: Identifier
    context_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    revision: int = Field(ge=1)
    session_id: Identifier
    phase: Literal["initial", "feedback", "final_selection"]
    history: list = Field(max_length=4)
    history_state: dict = Field(default_factory=dict)
    task: dict
    user_request: dict
    dataset_profile: dict
    knowledge: dict
    capabilities: dict
    evaluation: dict
    budget: dict
    evidence_ids: list[Identifier]
    proposal_schema: dict
    submission_guide: dict = Field(default_factory=dict)


def execution_spec(alternative):
    plan = alternative.optimization.model_dump(exclude={"objective"})
    plan["fixed_params"] = alternative.pipeline.fixed_params
    return {"method": alternative.pipeline.model_method, "steps": [step.model_dump() for step in alternative.pipeline.preprocessing], "max_components": 20, "parameter_plan": plan}
