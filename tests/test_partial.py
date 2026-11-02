"""Exhaustive software-policy checks; no astrophysical utility is established."""
from itertools import permutations, product

import numpy as np
import pytest

from applecider_agents.core.evidence import EvidenceScope, ToolObservation
from applecider_agents.core.partial import Decision, FinitePolicy, evaluate_partial
from applecider_agents.core.report import Report
from applecider_agents.flare.partial import (
    GROUPS, build_policy, evaluate_flare, legacy_success_decision, map_observations, nullable_legacy_decision, report_source_complete,
)
from applecider_agents.flare.partial import _evaluate_flare, _flare_reachability

SCOPE = EvidenceScope(obj_id="fixture", horizon_days=10.0, session_id="snapshot")


def capture(tool, payload, **updates):
    return ToolObservation(**(dict(tool=tool, scope=SCOPE, inputs={"obj_id": "fixture", "horizon_days": 10.0},
                                  status="success", payload=payload) | updates))


def measurements():
    return {
        "catalog_lookup": {"already_classified": False},
        "get_photometry": {"n_detections": 20, "peak_mag_observed": 18.0, "span_days": 10.0, "bands": {"g": {"peak_day": 3.0}}},
        "flare_classify": {"model": "bts6_h10", "predicted": "SN_Ia", "probabilities": {"SN_Ia": .9, "SN_CC": .1},
                           "prediction_set": ["SN_Ia"], "credibility": .8, "argmax_excluded_from_set": False},
        "flare_anomaly": {"alarm_score": "consensus", "alarm_p": .5, "alarm_far": None, "alarms": {"1%": False, "5%": False, "10%": False},
                          "calibrated": True, "calibration": "full", "energy_percentile": 0.0},
        "get_context": {"flags": dict(stellar_astrometry=False, nuclear=False, agn_mid_ir_colour=False,
                                      superluminous_if_host_is_right=False, offset_from_host_gt_10arcsec=False),
                        "host": {"offset_arcsec": 2.0}, "redshift": {"z_phot": None, "M_pseudo": None},
                        "gaia": {"point_source_within_2arcsec": False}},
    }


def observations(payloads=None):
    return [capture(tool, payload) for tool, payload in (measurements() if payloads is None else payloads).items()]


def independent_legacy_boolean_reference(w, blind):
    # Independently transcribed branch assignments from frozen upstream mock_triage.
    span = 401 if w["long_history"] and (w["class_history_trigger"] or w["any_alarm"]) else 0
    if w["catalogued"] and not blind:
        result = "already_classified", 0
    elif not w["enough_detections"]:
        result = "insufficient_data", 0
    elif w["stellar"] or (w["nuclear"] and span > 400) or (w["agn_colour"] and span > 400):
        result = "likely_stellar_or_agn", 1 if w["alarm_hit"] else 0
    elif w["alarm_hit"] or w["class_anomaly"]:
        result = "anomaly_review", 2 if (w["young"] or w["luminous"]) else 1
    elif w["rare_prediction"] or (w["set_tde"] and w["nuclear"]) or (w["set_slsn"] and w["luminous"] and not w["far_host"]):
        result = "needs_spectrum", 3 if (w["young"] and w["bright"]) else 2
    else:
        result = "ordinary", 0
    return Decision(*result)


@pytest.mark.parametrize("blind", [False, True])
def test_exhaustive_full_domain_matches_independent_reference(blind):
    policy = build_policy(blind)
    assert len(policy.states) == 131072
    assert policy.domain_kind == "conservative_overapproximation"
    for w, expected in zip(policy.worlds(), policy.decisions):
        assert independent_legacy_boolean_reference(w, blind) == expected
        assert legacy_success_decision(w, blind=blind) == expected
        assert nullable_legacy_decision(w, blind=blind) == expected


def toy_policy():
    worlds = [dict(zip(("a", "b", "c"), values)) for values in product((False, True), repeat=3)]
    return FinitePolicy.from_worlds("xor-with-irrelevant", worlds,
        lambda w: Decision("needs_spectrum" if w["a"] ^ w["b"] else "ordinary", 0),
        predicates=("a", "b", "c"), tool_predicates={"joint": ("a", "b"), "irrelevant": ("c",)})


def test_exhaustive_partial_soundness_monotonicity_and_arrival_order():
    policy = toy_policy()
    for values in product((None, False, True), repeat=3):
        known = {k: v for k, v in zip(policy.predicates, values) if v is not None}
        r = evaluate_partial(policy, known)
        expected = {d for w, d in zip(policy.worlds(), policy.decisions) if all(w[k] == v for k, v in known.items())}
        assert set(r.possible_decisions) == expected
        assert r.status == ("decided" if len(expected) == 1 else "unresolved")
        for ordering in permutations(known):
            assert evaluate_partial(policy, {k: known[k] for k in ordering}) == r
        for key in set(policy.predicates) - set(known):
            for v in (False, True):
                child = evaluate_partial(policy, known | {key: v})
                assert set(child.possible_decisions) <= expected


def test_correlated_xor_does_not_lose_joint_query_dependencies():
    states = np.array([[0,0,0,0], [0,0,1,1], [1,1,0,0], [1,1,1,1]], dtype=bool)
    policy = FinitePolicy("correlated-xor", ("a", "b", "c", "d"), states,
                          [Decision(str(v), 0) for v in (0,1,1,0)], {"first": ("a","b"), "second": ("c","d")})
    result = evaluate_partial(policy, {})
    assert result.status == "unresolved"
    assert set(result.missing_predicates) == set(policy.predicates)
    assert evaluate_partial(policy, {}, include_missing=False).missing_predicates == ()


def test_empty_support_is_inconsistent_and_empirical_zero_does_not_remove_world():
    policy = toy_policy()
    empirical_weights = [1.0 if d.verdict == "ordinary" else 0.0 for d in policy.decisions]
    assert sum(empirical_weights) > 0
    assert len(evaluate_partial(policy, {}).possible_decisions) == 2
    empty = FinitePolicy("empty", ("a",), np.zeros((0,1), bool), (), {"tool": ("a",)})
    r = evaluate_partial(empty, {})
    assert r.status == "inconsistent" and r.possible_decisions == ()
    assert evaluate_partial(policy, {}, conflicts=("incompatible-observations",)).status == "inconsistent"


def test_generic_domain_validation():
    with pytest.raises(ValueError):
        evaluate_partial(toy_policy(), {"unknown": True})
    with pytest.raises(ValueError):
        evaluate_partial(toy_policy(), {"a": 1})
    with pytest.raises(ValueError):
        FinitePolicy("bad", ("a",), np.array([[0]]), [Decision("ordinary",0)], {"tool": ("a",)})
    with pytest.raises(ValueError):
        FinitePolicy("conflict", ("a",), np.array([[True],[True]]), [Decision("ordinary",0),Decision("needs_spectrum",2)], {"tool": ("a",)})


def test_constant_policy_with_no_predicates():
    p = FinitePolicy.from_worlds("constant", [{}], lambda w: Decision("ordinary",0), predicates=(), tool_predicates={})
    assert evaluate_partial(p, {}).possible_decisions == (Decision("ordinary",0),)


def test_report_profile_cannot_be_substituted_for_decision_only():
    obs = [capture("catalog_lookup", {"already_classified": True})]
    decision = evaluate_flare(SCOPE, obs)
    report = evaluate_flare(SCOPE, obs, profile="legacy_report")
    assert decision.decision == Decision("already_classified",0) and decision.task_complete
    assert report.decision_certified and not report.report_complete and not report.task_complete
    assert report.report is None and "energy_percentile" in report.missing_report_fields
    assert not isinstance(report, Report) and not hasattr(report, "comment") and not hasattr(report, "verdict")


def test_complete_report_uses_measured_zero_and_explicit_missing_context():
    r = evaluate_flare(SCOPE, observations(), profile="legacy_report")
    assert r.status == "decided" and r.decision == Decision("ordinary",0)
    assert r.report_complete and r.task_complete
    assert r.report["energy_percentile"] == 0.0
    assert any("M_pseudo unavailable" in e["fact"] for e in r.report["evidence"])
    assert any("photo-z" in c for c in r.report["caveats"])
    assert r.metadata["prose_semantically_verified"] is False
    assert "gaia.parallax_over_error" in r.metadata["context_raw_missing"]
    # The frozen target trusts producer-computed False, without pretending its
    # raw scientific inputs were measured negatives.
    assert map_observations(SCOPE, observations()).constraints["stellar"] is False
    assert any("upstream missing raw" in caveat for caveat in r.report["caveats"])


@pytest.mark.parametrize("tool,key", [("flare_classify","credibility"), ("flare_classify","probabilities"),
                                      ("flare_anomaly","energy_percentile"), ("flare_anomaly","calibration"),
                                      ("flare_anomaly","alarms"), ("get_context","host")])
def test_missing_mandatory_fields_never_render_complete_report(tool,key):
    payloads = measurements()
    del payloads[tool][key]
    r = evaluate_flare(SCOPE, observations(payloads), profile="legacy_report")
    assert not r.report_complete and r.report is None
    assert not report_source_complete(map_observations(SCOPE, observations(payloads)), tool)


def test_report_source_validity_is_separate_from_predicates_and_trigger():
    payloads = measurements()
    before = map_observations(SCOPE, observations(payloads))
    del payloads["flare_anomaly"]["energy_percentile"]
    after = map_observations(SCOPE, observations(payloads))
    assert before.constraints == after.constraints
    assert report_source_complete(before, "flare_anomaly")
    assert not report_source_complete(after, "flare_anomaly")
    assert not report_source_complete(before, "light_curve_history")  # trigger false is not measured history


@pytest.mark.parametrize("key,value", [
    ("predicted", None), ("predicted", []), ("predicted", {}),
    ("prediction_set", None), ("prediction_set", [None]), ("prediction_set", [{}]),
    ("prediction_set", ["SN_Ia", "SN_Ia"]), ("probabilities", None),
    ("probabilities", []), ("probabilities", {"SN_Ia": []}),
])
def test_malformed_nested_classification_fields_remain_unknown_without_crash(key,value):
    payloads = measurements()
    payloads["flare_classify"][key] = value
    obs = observations(payloads)
    result = evaluate_flare(SCOPE, obs, profile="legacy_report")
    assert not result.report_complete and result.report is None
    mapped = map_observations(SCOPE, obs)
    if key == "prediction_set":
        assert "set_tde" not in mapped.constraints and "set_slsn" not in mapped.constraints
    if key == "predicted":
        assert "rare_prediction" not in mapped.constraints


@pytest.mark.parametrize("block,key", [("host","offset_arcsec"),("redshift","z_phot"),("redshift","M_pseudo"),("gaia","point_source_within_2arcsec")])
def test_malformed_present_context_is_never_rendered_as_measured_science(block,key):
    payloads = measurements()
    payloads["get_context"][block][key] = "INVALID_NUMERIC"
    result = evaluate_flare(SCOPE, observations(payloads), profile="legacy_report")
    assert result.decision_certified  # successful explicit policy flags are a separate input
    assert not result.report_complete and result.report is None
    assert f"{block}.{key}" in result.metadata["context_invalid_fields"]


@pytest.mark.parametrize("tool", ["get_context","get_photometry","flare_classify","flare_anomaly"])
def test_omitted_horizon_cannot_be_relabelled_as_an_early_prefix(tool):
    payload = measurements()[tool]
    no_horizon = capture(tool, payload, inputs={"obj_id":"fixture"})
    mapped = map_observations(SCOPE,[no_horizon])
    assert not mapped.constraints and tool in mapped.issues
    explicit_import = capture(tool,payload,inputs={"obj_id":"fixture"},provenance={"relative_photometric_horizon_days":10.0})
    assert map_observations(SCOPE,[explicit_import]).constraints


def test_frozen_truthiness_count_only_and_conditional_history():
    payloads = measurements()
    payloads["flare_classify"]["credibility"] = 0.0
    payloads["get_photometry"]["peak_mag_observed"] = 0.0
    payloads["get_photometry"]["bands"] = {}  # legacy has no per-band detection floor
    mapped = map_observations(SCOPE, observations(payloads))
    assert mapped.constraints["class_history_trigger"] is True
    assert mapped.constraints["class_anomaly"] is False
    assert mapped.constraints["bright"] is False
    assert mapped.constraints["enough_detections"] is True
    r = evaluate_flare(SCOPE, observations(payloads), profile="legacy_report")
    assert r.decision == Decision("ordinary",0) and not r.report_complete
    assert any("history" in field for field in r.missing_report_fields)


def test_unavailable_history_is_unknown_and_completed_negative_is_distinct():
    payloads = measurements()
    payloads["flare_classify"]["prediction_set"] = ["SN_Ia", "AGN"]
    payloads["get_context"]["flags"]["nuclear"] = True
    obs = observations(payloads)
    r = evaluate_flare(SCOPE, obs + [capture("light_curve_history", None, status="unavailable")], profile="legacy_report")
    assert r.status == "unresolved" and not r.task_complete
    unverified_empty = capture("light_curve_history", {"found": False})
    assert "long_history" not in map_observations(SCOPE, [unverified_empty]).constraints
    complete_empty = capture("light_curve_history", {"found": False}, provenance={"completed_search": True})
    r = evaluate_flare(SCOPE, obs + [complete_empty], profile="legacy_report")
    assert r.decision == Decision("ordinary",0) and r.report_complete
    assert any("completed history search" in x["fact"] for x in r.report["evidence"])
    measured = capture("light_curve_history", {"found": True, "n_detections": 25, "activity_span_days": 500.0})
    assert evaluate_flare(SCOPE, obs + [measured]).decision == Decision("likely_stellar_or_agn",0)


@pytest.mark.parametrize("fault", ["error", "nan", "wrong_object", "wrong_horizon", "stale_session", "missing"])
def test_faulty_observation_cannot_supply_false_decision(fault):
    payload = {"already_classified": False}
    kwargs = {}
    if fault == "error": kwargs["status"] = "error"
    if fault == "nan": payload["value"] = float("nan")
    if fault == "wrong_object": payload["obj_id"] = "other"
    if fault == "wrong_horizon": kwargs["inputs"] = {"obj_id": "fixture", "horizon_days": 100}
    if fault == "stale_session": kwargs["scope"] = SCOPE.model_copy(update={"session_id": "old"})
    if fault == "missing": payload = {}
    obs = [o for o in observations() if o.tool != "catalog_lookup"] + [capture("catalog_lookup",payload,**kwargs)]
    r = evaluate_flare(SCOPE,obs)
    assert r.status == "unresolved" and Decision("already_classified",0) in r.possible_decisions


def test_failures_do_not_reproduce_literal_classification_error_fallback():
    obs = [capture("flare_classify", {"error": "no input"}), capture("catalog_lookup", {"already_classified": True})]
    r = evaluate_flare(SCOPE, obs)
    assert r.decision == Decision("already_classified",0)
    assert "flare_classify" in r.rejected_evidence
    assert evaluate_flare(SCOPE, [obs[0]]).status == "unresolved"


def test_conflicting_duplicate_observations_inconsistent_in_every_order():
    a = capture("catalog_lookup", {"already_classified": True})
    b = capture("catalog_lookup", {"already_classified": False})
    for order in ((a,b),(b,a)):
        result = evaluate_flare(SCOPE, order)
        assert result.status == "inconsistent" and not result.decision_certified and result.report is None


def test_complete_tool_arrival_order_and_snapshot_replacement():
    obs = observations()
    baseline = evaluate_flare(SCOPE,obs)
    for ordering in permutations(obs):
        r = evaluate_flare(SCOPE, ordering)
        assert r.possible_decisions == baseline.possible_decisions and r.status == baseline.status
    fresh = SCOPE.model_copy(update={"session_id":"replacement"})
    assert evaluate_flare(fresh,obs).status == "unresolved"


def test_unknown_profile_and_blind_catalogue_configuration():
    with pytest.raises(ValueError):
        evaluate_flare(SCOPE, [], profile="invented")
    obs = observations()
    obs.append(capture("catalog_lookup", {"already_classified": True}))
    result = evaluate_flare(SCOPE,obs,blind=True)
    assert result.decision == Decision("ordinary",0)
    assert "catalog_lookup" not in result.used_evidence.get("catalogued",[])


@pytest.mark.parametrize("blind", [False, True])
def test_fastpath_full_abstract_domain_without_enumeration(blind, monkeypatch):
    import applecider_agents.flare.partial as adapter
    policy = build_policy(blind)
    def unexpected_enumeration(*args, **kwargs):
        raise AssertionError("complete nullable decisions must not construct a policy table")
    monkeypatch.setattr(adapter, "build_policy", unexpected_enumeration)
    for world, decision in zip(policy.worlds(), policy.decisions):
        result = _flare_reachability(world, blind=blind)
        assert result.possible_decisions == (decision,)
        assert result.admissible_count == 1 and result.missing_predicates == ()


def test_fastpath_is_lazy_and_conflicts_take_precedence(monkeypatch):
    import applecider_agents.flare.partial as adapter
    def unexpected_enumeration(*args, **kwargs):
        raise AssertionError("certified singleton or conflict must not allocate the table")
    build_policy.cache_clear()
    monkeypatch.setattr(adapter, "build_policy", unexpected_enumeration)
    result = evaluate_flare(SCOPE, [capture("catalog_lookup", {"already_classified": True})])
    assert result.decision == Decision("already_classified",0)
    assert result.metadata["admissible_count"] == 2**16
    conflicting = [capture("catalog_lookup", {"already_classified": v}) for v in (False,True)]
    result = evaluate_flare(SCOPE,conflicting)
    assert result.status == "inconsistent" and result.metadata["admissible_count"] == 0


@pytest.mark.parametrize("blind", [False, True])
def test_fastpath_partial_constraint_equivalence_and_missing_option(blind):
    rng = np.random.default_rng(20261102)
    policy = build_policy(blind)
    for _ in range(128):
        values = rng.integers(-1,2,size=len(policy.predicates))
        constraints = {p: bool(v) for p,v in zip(policy.predicates,values) if v >= 0}
        for include_missing in (False,True):
            fast = _flare_reachability(constraints,blind=blind,include_missing=include_missing)
            exact = _flare_reachability(constraints,blind=blind,include_missing=include_missing,_force_enumeration=True)
            assert fast == exact


@pytest.mark.parametrize("profile", ["decision_only", "legacy_report"])
def test_fastpath_complete_serialized_outputs_for_all_tool_subsets(profile):
    obs = observations()
    for mask in range(1 << len(obs)):
        subset = [o for i,o in enumerate(obs) if mask & (1 << i)]
        for blind in (False,True):
            fast = evaluate_flare(SCOPE,subset,profile=profile,blind=blind)
            exact = _evaluate_flare(SCOPE,subset,profile=profile,blind=blind,_force_enumeration=True)
            assert fast.model_dump(mode="json") == exact.model_dump(mode="json")


def test_fastpath_malformed_and_conflicting_output_equivalence():
    cases = [[capture("catalog_lookup", {"already_classified": v}) for v in (True,False)],
             [capture("catalog_lookup", {"already_classified": True},scope=SCOPE.model_copy(update={"session_id":"stale"}))]]
    for field, value in (("credibility",float("nan")),("predicted",[]),("prediction_set",["SN_Ia","SN_Ia"])):
        payloads = measurements()
        payloads["flare_classify"][field] = value
        cases.append(observations(payloads))
    payloads = measurements()
    payloads["get_context"]["host"]["offset_arcsec"] = "INVALID_NUMERIC"
    cases.append(observations(payloads))
    for obs in cases:
        for profile in ("decision_only","legacy_report"):
            for blind in (False,True):
                assert evaluate_flare(SCOPE,obs,profile=profile,blind=blind).model_dump(mode="json") == _evaluate_flare(
                    SCOPE,obs,profile=profile,blind=blind,_force_enumeration=True).model_dump(mode="json")
