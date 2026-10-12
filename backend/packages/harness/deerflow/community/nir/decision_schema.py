"""Phase-specific provider views of the existing strict decision contract.

Views guide syntax and process order, never select a pipeline or an action.
The original receiver still validates exact model-issued arguments and budgets.
"""

import copy

from .decision_contracts import DecisionProposal


def proposal_schema(phase):
    schema = DecisionProposal.model_json_schema()
    properties = schema["properties"]
    actions = {
        "initial": ["EXPLORE"],
        "feedback": ["TUNE", "CHANGE_PREPROCESS", "SWITCH_MODEL", "EXPLORE", "STOP"],
        "final_selection": ["RETAIN"],
    }[phase]
    properties["phase"] = {"type": "string", "const": phase}
    properties["action"] = {"type": "string", "enum": actions}
    alternative = schema["$defs"]["Alternative"]
    alternative["properties"]["action"] = {"type": "string", "enum": actions}
    alternative["required"] = list(dict.fromkeys([*alternative["required"], "action"]))
    if phase != "initial":
        schema["required"] += ["base_experiment_id", "uncertainties", "hypotheses", "not_selected_reasons", "expected_fits"]
        for key in ("uncertainties", "hypotheses"):
            properties[key]["minItems"] = 1
        properties["not_selected_reasons"]["minProperties"] = 1
    if phase == "final_selection":
        schema["required"] += ["retained_experiment_id", "stop_evidence"]
        properties["expected_fits"] = {"type": "integer", "const": 0}
        for key in ("pipeline", "optimization"):
            alternative["properties"][key] = {"type": "null"}
    # Conditional constraints express the same terminal/feedback checks enforced
    # in submit(). No rejected plan is repaired or replaced by this view.
    if phase == "feedback":
        schema["allOf"] = [{"if": {"properties": {"action": {"const": "STOP"}}}, "then": {"required": ["retained_experiment_id", "stop_evidence"], "properties": {"expected_fits": {"const": 0}}}}]
    alternative["allOf"] = [
        {
            "if": {"properties": {"action": {"enum": ["STOP", "RETAIN"]}}},
            "then": {"properties": {"pipeline": {"type": "null"}, "optimization": {"type": "null"}}},
            "else": {"properties": {"pipeline": {"$ref": "#/$defs/PipelineSpec"}, "optimization": {"$ref": "#/$defs/SearchSpec"}}},
        }
    ]
    if phase == "final_selection":
        alternative.pop("allOf")
    else:
        from nir_core.model.parameter_control import SCHEMAS

        for method, parameters in SCHEMAS.items():
            writable = {}
            for name, spec in parameters.items():
                value = {"enum": spec["values"]} if spec["type"] == "enum" else {key: spec[key] for key in ("type", "minimum", "maximum")}
                options = [value]
                if spec.get("strings"):
                    options.append({"type": "string", "enum": spec["strings"]})
                if spec.get("nullable"):
                    options.append({"type": "null"})
                writable[name] = {"anyOf": options} if len(options) > 1 else value
            condition = {"properties": {"pipeline": {"type": "object", "properties": {"model_method": {"const": method}}, "required": ["model_method"]}}, "required": ["pipeline"]}
            fixed = {"type": "object", "properties": writable, "additionalProperties": False}
            space = {"type": "object", "properties": {name: {"type": "array", "items": value, "minItems": 1, "maxItems": 20} for name, value in writable.items()}, "additionalProperties": False}
            alternative["allOf"].append({"if": condition, "then": {"properties": {"pipeline": {"properties": {"fixed_params": fixed}}, "optimization": {"properties": {"search_space": space}}}}})
    # Discard unreachable definitions (e.g. all training schemas at RETAIN).
    definitions = schema.pop("$defs")
    reachable = set()

    def visit(value):
        if isinstance(value, dict):
            ref = value.get("$ref", "")
            if ref.startswith("#/$defs/"):
                key = ref.split("/")[-1]
                if key not in reachable:
                    reachable.add(key)
                    visit(definitions[key])
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(schema)
    schema["$defs"] = {key: value for key, value in definitions.items() if key in reachable}

    def compact(value):
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items() if key not in {"title", "default"}}
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value

    return compact(schema)


def submission_guide(phase, folds):
    guide = {
        "submit_tool": "nir_submit_decision",
        "proposal_format": "JSON object; old JSON strings remain accepted",
        "read_only": ["dataset_profile", "history", "capabilities", "evaluation", "budget", "operational_parameters"],
        "binding_fields": {"context_id": "copy context_id", "context_sha256": "copy context_sha256", "expected_revision": "copy revision"},
        "nesting": (
            "alternatives[i].pipeline holds preprocessing/model_method/fixed_params; "
            "sibling alternatives[i].optimization holds strategy/search_space/max_trials/max_fits. "
            "Never nest optimization inside pipeline or fixed_params inside optimization."
        ),
        "choice_consistency": "action must equal the selected alternative action; not_selected_reasons maps EVERY other alternative id to a reason. All observations must cite evidence_ids.",
        "feedback_required": ["base_experiment_id", "uncertainties", "hypotheses", "not_selected_reasons", "expected_fits"] if phase != "initial" else [],
        "syntax_fragments_not_recommendations": {
            "fixed_optimization": {"strategy": "fixed", "search_space": {}, "max_trials": 1, "max_fits": folds + 1},
            "TUNE": {"action": "TUNE", "difference": "Change parameters; keep the observed model and preprocessing"},
            "CHANGE_PREPROCESS": {"action": "CHANGE_PREPROCESS", "difference": "Change preprocessing; keep the observed model"},
            "SWITCH_MODEL": {"action": "SWITCH_MODEL", "difference": "Change model; keep the observed preprocessing"},
            "EXPLORE": {"action": "EXPLORE", "difference": "Explain the new direction compared with persisted experiments"},
            "STOP" if phase != "final_selection" else "RETAIN": {"action": "STOP" if phase != "final_selection" else "RETAIN", "expected_fits": 0, "selected_alternative_fields": {"pipeline": None, "optimization": None}},
        },
        "terminal_requirements": "Name a successful retained_experiment_id, cite experiment:<id>, include stop_evidence advantages/limitations/coverage/unverified directions. No training plan or fit is executed.",
        "reuse": "Reuse this immutable context for correction. Read a new context only after execution or a reported stale context/budget transition.",
        "concise_submission": (
            "Keep observations, comparisons and scientific reasons brief to fit the per-response output cap. "
            "Two distinct compared alternatives suffice; only the selected one runs. "
            "expected_outcome is required at proposal root, never inside alternatives. "
            "Omit optional adapter defaults; do not omit required fields or evidence refs. "
            "No hidden reasoning transcript is requested."
        ),
    }
    allowed = {"initial": {"EXPLORE", "fixed_optimization"}, "feedback": {"TUNE", "CHANGE_PREPROCESS", "SWITCH_MODEL", "EXPLORE", "STOP", "fixed_optimization"}, "final_selection": {"RETAIN"}}[phase]
    guide["syntax_fragments_not_recommendations"] = {key: value for key, value in guide["syntax_fragments_not_recommendations"].items() if key in allowed}
    return guide


def bound_tool_schema(context):
    schema = copy.deepcopy(context["proposal_schema"])
    properties = schema["properties"]
    for key, value in (("context_id", context["context_id"]), ("context_sha256", context["context_sha256"]), ("expected_revision", context["revision"])):
        properties[key] = {"const": value, "type": "integer" if key == "expected_revision" else "string"}
    definitions = schema.pop("$defs")
    evidence_ids = context.get("evidence_ids")
    if evidence_ids:
        for fields in (properties, definitions["EvidenceClaim"]["properties"]):
            items = fields["evidence_refs"]["items"]
            fields["evidence_refs"]["items"] = {**items, "enum": list(evidence_ids)}
    return {"type": "object", "properties": {"proposal": schema}, "required": ["proposal"], "additionalProperties": False, "$defs": definitions}
