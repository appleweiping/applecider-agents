"""Observation ownership and scope are independent of policy outcomes."""
import json

import pytest
from pydantic import ValidationError

from applecider_agents.core.evidence import EvidenceScope, ToolObservation, finite_number, finite_science


def scope(**updates):
    return EvidenceScope(**(dict(obj_id="object", horizon_days=10.0, session_id="snapshot-1") | updates))


def observation(**updates):
    return ToolObservation(**(dict(tool="tool", scope=scope(), inputs={"obj_id": "object", "horizon_days": 10.0},
                                  status="success", payload={"obj_id": "object", "measured": 0.0}) | updates))


def test_captured_payload_owns_input_containers_and_round_trips():
    payload = {"obj_id": "object", "nested": {"values": [1, 2]}}
    inputs = {"obj_id": "object", "horizon_days": 10.0}
    provenance = {"source": ["fixture"]}
    obs = observation(payload=payload, inputs=inputs, provenance=provenance)
    payload["nested"]["values"].append(3)
    inputs["horizon_days"] = 20
    provenance["source"].append("changed")
    assert obs.payload["nested"]["values"] == [1, 2]
    assert obs.inputs["horizon_days"] == 10
    assert obs.provenance == {"source": ["fixture"]}
    assert ToolObservation.model_validate(json.loads(obs.model_dump_json())) == obs
    with pytest.raises(ValidationError):
        obs.status = "error"


@pytest.mark.parametrize("updates", [{"obj_id": "other"}, {"horizon_days": 20}, {"session_id": "snapshot-2"}])
def test_scope_mismatch_is_unknown(updates):
    assert observation().unusable_reason(scope(**updates))


@pytest.mark.parametrize("which,updates", [(which, updates) for which in ("inputs", "payload") for updates in
                         ({"obj_id": "other"}, {"horizon_days": 20}, {"horizon_days": float("nan")}, {"session_id": "other"})])
def test_actual_request_and_payload_bindings_checked(which, updates):
    kwargs = {which: {"obj_id": "object", "horizon_days": 10.0} | updates}
    assert observation(**kwargs).unusable_reason(scope())


@pytest.mark.parametrize("status", ["error", "unavailable", "not_required"])
def test_unsuccessful_record_cannot_be_used_as_false(status):
    assert observation(status=status, payload={"measured": False}).unusable_reason(scope())


@pytest.mark.parametrize("payload", [None, {"error": "failed", "measured": False}])
def test_missing_and_error_payloads_are_not_success(payload):
    assert observation(payload=payload).unusable_reason(scope())


def test_scope_declaration_does_not_replace_missing_actual_input_identity():
    assert observation(inputs={}).unusable_reason(scope())


def test_finite_values_preserve_real_zero_and_separate_boolean():
    assert finite_number(0.0) and finite_number(0)
    assert not finite_number(False) and not finite_number(None)
    assert not finite_science({"nested": [float("inf")]})
    assert finite_science({"missing": None, "measured": 0.0})


@pytest.mark.parametrize("horizon", [float("inf"), float("nan"), -1])
def test_scope_cannot_hold_invalid_horizon(horizon):
    with pytest.raises(ValidationError):
        scope(horizon_days=horizon)
