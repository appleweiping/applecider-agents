"""Opt-in evidence-aware evaluation of the frozen successful-data legacy policy.

The Boolean product below is deliberately a conservative overapproximation of
real tool outputs: it does not infer physical constraints from their empirical
frequency. A singleton is sound for that domain; unresolved sets need not be
maximal for the realizable astronomical domain. This is software-policy evidence,
not evidence that a telescope action is scientifically optimal.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, Field

from ..core.evidence import EvidenceScope, ToolObservation, finite_number, finite_science
from ..core.partial import Decision, FinitePolicy, PartialResult, Reachability, evaluate_partial

POLICY_ID = "legacy-v1-successful-data"
GROUPS = {
    "catalog_lookup": ("catalogued",),
    "get_photometry": ("enough_detections", "young", "bright"),
    "flare_classify": ("class_anomaly", "rare_prediction", "set_tde", "set_slsn", "class_history_trigger"),
    "flare_anomaly": ("alarm_hit", "any_alarm"),
    "get_context": ("stellar", "nuclear", "agn_colour", "luminous", "far_host"),
    "light_curve_history": ("long_history",),
}
_PREDICATES = tuple(p for names in GROUPS.values() for p in names)
_DOMAIN_KIND = "conservative_overapproximation"
_DOMAIN_NOTES = ("Independent Boolean threshold product; no claim that every compressed state is realizable. "
                 "Actual classifier/set and host-threshold correlations are intentionally not imposed.")
CLASSES = {"SN_Ia", "SN_CC", "SLSN", "AGN", "TDE", "CV"}
ACTIONS = {"needs_spectrum": "low-resolution spectrum (SEDM class) at next opportunity",
           "anomaly_review": "human look at cutouts and host; one more g and r epoch",
           "likely_stellar_or_agn": "check Gaia and the long-term light curve; no spectrum"}
SCOPE_LIMITATIONS = (
    "Photometric horizon means relative dt <= horizon only; absolute event epochs are unavailable.",
    "Static catalogue availability dates are unknown; this is not historical as-of or prospective evaluation.",
    "Full-history prefix alignment is not certified without aligned timestamped raw observations.",
    "Explicit context flags are legacy tool-computed policy inputs; upstream missing raw measurements may synthesize False. This adapter does not repair that scientific missingness.",
)


def legacy_success_decision(w: Mapping[str, bool], *, blind: bool = False) -> Decision:
    """Frozen branch/truthiness semantics after successful relevant measurements.

    class_anomaly includes `(credibility or 1) < .1`, so measured zero
    credibility alone does not activate that legacy anomaly branch. The history
    trigger uses `(credibility or 0) < .2`. The detection test has no per-band
    minimum. Catalogue precedence does not reproduce classify-error fallback.
    """
    history_long = (w["class_history_trigger"] or w["any_alarm"]) and w["long_history"]
    if not blind and w["catalogued"]:
        return Decision("already_classified", 0)
    if not w["enough_detections"]:
        return Decision("insufficient_data", 0)
    if w["stellar"] or ((w["nuclear"] or w["agn_colour"]) and history_long):
        return Decision("likely_stellar_or_agn", int(w["alarm_hit"]))
    if w["alarm_hit"] or w["class_anomaly"]:
        return Decision("anomaly_review", 2 if w["young"] or w["luminous"] else 1)
    if w["rare_prediction"] or (w["set_tde"] and w["nuclear"]) or (w["set_slsn"] and w["luminous"] and not w["far_host"]):
        return Decision("needs_spectrum", 3 if w["young"] and w["bright"] else 2)
    return Decision("ordinary", 0)


@lru_cache(maxsize=2)
def build_policy(blind: bool = False) -> FinitePolicy:
    predicates = _PREDICATES
    states = ((np.arange(1 << len(predicates), dtype=np.uint32)[:, None] >>
               np.arange(len(predicates), dtype=np.uint32)) & 1).astype(bool)
    b = dict(zip(predicates, states.T))
    # Vectorized independently from the scalar reference, retaining branch order.
    labels = np.zeros(len(states), dtype=np.int32)
    priority = np.zeros(len(states), dtype=np.int32)
    rare = b["rare_prediction"] | (b["set_tde"] & b["nuclear"]) | (b["set_slsn"] & b["luminous"] & ~b["far_host"])
    labels[rare] = 1
    priority[rare] = np.where(b["young"][rare] & b["bright"][rare], 3, 2)
    anomaly = b["alarm_hit"] | b["class_anomaly"]
    labels[anomaly] = 2
    priority[anomaly] = np.where(b["young"][anomaly] | b["luminous"][anomaly], 2, 1)
    persistent = (b["class_history_trigger"] | b["any_alarm"]) & b["long_history"]
    stellar = b["stellar"] | ((b["nuclear"] | b["agn_colour"]) & persistent)
    labels[stellar] = 3
    priority[stellar] = b["alarm_hit"][stellar].astype(int)
    labels[~b["enough_detections"]] = 4
    priority[~b["enough_detections"]] = 0
    if not blind:
        labels[b["catalogued"]] = 5
        priority[b["catalogued"]] = 0
    verdicts = ("ordinary", "needs_spectrum", "anomaly_review", "likely_stellar_or_agn", "insufficient_data", "already_classified")
    cache = {(v, p): Decision(v, p) for v in verdicts for p in range(4)}
    decisions = tuple(cache[(verdicts[int(v)], int(p))] for v, p in zip(labels, priority))
    groups = {k: v for k, v in GROUPS.items() if not (blind and k == "catalog_lookup")}
    return FinitePolicy(POLICY_ID + ("-blind" if blind else ""), predicates, states, decisions, groups,
                        _DOMAIN_KIND, _DOMAIN_NOTES)


class MappingResult(BaseModel):
    constraints: dict[str, bool] = Field(default_factory=dict)
    sources: dict[str, list[str]] = Field(default_factory=dict)
    payloads: dict[str, dict[str, Any]] = Field(default_factory=dict)
    issues: dict[str, list[str]] = Field(default_factory=dict)
    usable_tools: tuple[str, ...] = ()
    valid_tools: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    observations: dict[str, dict[str, Any]] = Field(default_factory=dict)


def _or(*values):
    return True if any(v is True for v in values) else None if any(v is None for v in values) else False


def _and(*values):
    return False if any(v is False for v in values) else None if any(v is None for v in values) else True


def _bool(value):
    return value if type(value) is bool else None


def _number(value, low=None, high=None):
    return finite_number(value) and (low is None or value >= low) and (high is None or value <= high)


def _set(value):
    return (isinstance(value, (list, tuple)) and all(isinstance(v, str) and v in CLASSES for v in value)
            and len(set(value)) == len(value))


def _predicates(tool: str, p: Mapping[str, Any], obs: ToolObservation) -> dict[str, bool]:
    out = {}
    if tool == "catalog_lookup":
        out["catalogued"] = _bool(p.get("already_classified"))
    elif tool == "get_photometry":
        n = p.get("n_detections")
        out["enough_detections"] = n >= 8 if _number(n, 0) and int(n) == n else None
        mag = p.get("peak_mag_observed")
        out["bright"] = (mag or 99) < 19 if _number(mag) else None
        bands, span = p.get("bands"), p.get("span_days")
        if isinstance(bands, dict) and not bands:
            out["young"] = False
        elif isinstance(bands, dict) and _number(span, 0) and all(isinstance(b, dict) and "peak_day" in b and
                (b["peak_day"] is None or _number(b["peak_day"])) for b in bands.values()):
            out["young"] = any(b["peak_day"] is not None and span - b["peak_day"] < 10 for b in bands.values())
    elif tool == "flare_classify":
        cred, pred, prediction_set = p.get("credibility"), p.get("predicted"), p.get("prediction_set")
        low = (cred or 1) < .1 if _number(cred, 0, 1) else None
        out["class_anomaly"] = _or(_bool(p.get("argmax_excluded_from_set")), low)
        out["rare_prediction"] = pred in ("SLSN", "TDE") if isinstance(pred, str) and pred in CLASSES else None
        if _set(prediction_set):
            out["set_tde"], out["set_slsn"] = "TDE" in prediction_set, "SLSN" in prediction_set
        out["class_history_trigger"] = _or((cred or 0) < .2 if _number(cred, 0, 1) else None,
                                            len(prediction_set) != 1 if _set(prediction_set) else None)
    elif tool == "flare_anomaly":
        far = p.get("alarm_far")
        out["any_alarm"] = far is not None if "alarm_far" in p and (far is None or _number(far, 0, 1)) else None
        alarms = p.get("alarms")
        hit = _bool(alarms["1%"]) if isinstance(alarms, dict) and "1%" in alarms else _bool(p.get("above_1pct_far"))
        out["alarm_hit"] = _and(_bool(p.get("calibrated")), hit)
    elif tool == "get_context":
        flags = p.get("flags")
        if isinstance(flags, dict):
            out = {name: _bool(flags.get(key)) for name, key in zip(GROUPS[tool],
                   ("stellar_astrometry", "nuclear", "agn_mid_ir_colour", "superluminous_if_host_is_right", "offset_from_host_gt_10arcsec"))}
    elif tool == "light_curve_history":
        if p.get("found") is False and obs.provenance.get("completed_search") is True:
            out["long_history"] = False
        elif p.get("found") is True and _number(p.get("activity_span_days"), 0):
            out["long_history"] = p["activity_span_days"] > 400
    return {k: bool(v) for k, v in out.items() if v is not None}


def map_observations(scope: EvidenceScope, observations: Sequence[ToolObservation], *, blind: bool = False) -> MappingResult:
    result = MappingResult()
    conflicts = []
    for obs in observations:
        if obs.tool not in GROUPS or (blind and obs.tool == "catalog_lookup"):
            continue
        reason = obs.unusable_reason(scope)
        if reason is None and obs.tool in ("get_context", "get_photometry", "flare_classify", "flare_anomaly") and "horizon_days" not in obs.inputs:
            effective = obs.provenance.get("relative_photometric_horizon_days", 100.0)
            if not _number(effective, 0) or effective != scope.horizon_days:
                reason = "omitted tool horizon resolves to a different effective prefix (canonical default100 days)"
        if reason is None and not finite_science(obs.payload):
            reason = "non-finite scientific value in payload"
        if reason is not None:
            result.issues.setdefault(obs.tool, []).append(reason)
            continue
        p = obs.payload
        if obs.tool in result.payloads and result.payloads[obs.tool] != p:
            conflicts.append(f"{obs.tool}: unequal payloads in one snapshot")
        result.payloads[obs.tool] = p
        result.observations[obs.tool] = obs.model_dump(mode="json")
        known = _predicates(obs.tool, p, obs)
        for key, value in known.items():
            if key in result.constraints and result.constraints[key] != value:
                conflicts.append(key)
            result.constraints[key] = value
            result.sources[key] = [obs.tool]
        absent = set(GROUPS[obs.tool]) - set(known)
        if absent:
            result.issues.setdefault(obs.tool, []).append("missing or invalid predicate sources: " + ", ".join(sorted(absent)))
        if obs.tool == "get_context" and _context_invalid(p):
            result.issues.setdefault(obs.tool, []).append("malformed context scientific fields: " + ", ".join(_context_invalid(p)))
    result.usable_tools = tuple(sorted(result.payloads))
    result.valid_tools = tuple(sorted(t for t in result.payloads if all(p in result.constraints for p in GROUPS[t])))
    result.conflicts = tuple(sorted(set(conflicts)))
    return result


def nullable_legacy_decision(c: Mapping[str, bool], *, blind: bool = False) -> Decision | None:
    """Simple conservative direct-rule comparator; no possible-world search."""
    if not blind:
        cat = c.get("catalogued")
        if cat is None:
            return None
        if cat:
            return Decision("already_classified", 0)
    enough = c.get("enough_detections")
    if enough is None:
        return None
    if not enough:
        return Decision("insufficient_data", 0)
    persistent = _and(_or(c.get("nuclear"), c.get("agn_colour")),
                      _or(c.get("class_history_trigger"), c.get("any_alarm")), c.get("long_history"))
    stellar = _or(c.get("stellar"), persistent)
    if stellar is None:
        return None
    if stellar:
        return Decision("likely_stellar_or_agn", int(c["alarm_hit"])) if "alarm_hit" in c else None
    anomaly = _or(c.get("alarm_hit"), c.get("class_anomaly"))
    if anomaly is None:
        return None
    if anomaly:
        high = _or(c.get("young"), c.get("luminous"))
        return Decision("anomaly_review", 2 if high else 1) if high is not None else None
    rare = _or(c.get("rare_prediction"), _and(c.get("set_tde"), c.get("nuclear")),
               _and(c.get("set_slsn"), c.get("luminous"), None if "far_host" not in c else not c["far_host"]))
    if rare is None:
        return None
    if rare:
        high = _and(c.get("young"), c.get("bright"))
        return Decision("needs_spectrum", 3 if high else 2) if high is not None else None
    return Decision("ordinary", 0)


def _report_missing(mapped: MappingResult) -> list[str]:
    p, c = mapped.payloads, mapped.constraints
    missing = []
    cl, an, cx = (p.get(t, {}) for t in ("flare_classify", "flare_anomaly", "get_context"))
    pred = cl.get("predicted")
    for name, valid in (
        ("flare_label", isinstance(pred, str) and pred in CLASSES),
        ("prediction_set", _set(cl.get("prediction_set"))),
        ("credibility", _number(cl.get("credibility"), 0, 1)),
        ("classify.model", isinstance(cl.get("model"), str) and bool(cl["model"])),
        ("classify.predicted_probability", isinstance(pred, str) and isinstance(cl.get("probabilities"), dict) and _number(cl["probabilities"].get(pred), 0, 1)),
        ("energy_percentile", _number(an.get("energy_percentile"), 0, 100)),
        ("anomaly.alarm_score", isinstance(an.get("alarm_score"), str) and bool(an["alarm_score"])),
        ("anomaly.alarm_p", _number(an.get("alarm_p"), 0, 1)),
        ("anomaly.alarm_far", "alarm_far" in an and (an["alarm_far"] is None or _number(an["alarm_far"], 0, 1))),
        ("anomaly.alarms", isinstance(an.get("alarms"), dict) and all(type(an["alarms"].get(k)) is bool for k in ("1%", "5%", "10%"))),
        ("anomaly.calibrated", type(an.get("calibrated")) is bool),
        ("anomaly.calibration", isinstance(an.get("calibration"), str) and bool(an["calibration"])),
        ("context.host_eligibility", "host" in cx and (cx["host"] is None or isinstance(cx["host"], dict))),
        ("context.scientific_field_types", not _context_invalid(cx)),
    ):
        if not valid:
            missing.append(name)
    required = _or(c.get("class_history_trigger"), c.get("any_alarm"))
    if required is None:
        missing.append("history.trigger")
    elif required and not _history_complete(mapped):
        missing.append("history.completed_search_or_measured_span_and_count")
    return missing


def _history_complete(mapped: MappingResult) -> bool:
    history = mapped.payloads.get("light_curve_history", {})
    provenance = mapped.observations.get("light_curve_history", {}).get("provenance", {})
    if history.get("found") is False:
        return provenance.get("completed_search") is True
    return (history.get("found") is True and _number(history.get("activity_span_days"), 0)
            and _number(history.get("n_detections"), 0)
            and int(history["n_detections"]) == history["n_detections"])


def report_source_complete(mapped: MappingResult, tool: str) -> bool:
    """Whether this bound payload supplies its declared report-template fields.

    Independent of whether the tool is currently decision-relevant or history
    is triggered. Planners may carry this validity bit separately from predicate
    values; equal predicate tuples do not imply equal report completeness.
    """
    if tool not in GROUPS:
        raise ValueError("unknown tool")
    if tool not in mapped.payloads or mapped.conflicts:
        return False
    if tool == "light_curve_history":
        return _history_complete(mapped)
    missing = _report_missing(mapped)
    if tool == "flare_classify":
        return not any(p in ("flare_label", "prediction_set", "credibility") or p.startswith("classify.") for p in missing)
    if tool == "flare_anomaly":
        return not any(p == "energy_percentile" or p.startswith("anomaly.") for p in missing)
    if tool == "get_context":
        return not any(p.startswith("context.") for p in missing)
    return all(p in mapped.constraints for p in GROUPS[tool])


def _render(scope: EvidenceScope, decision: Decision, mapped: MappingResult) -> dict[str, Any]:
    cl, an, cx = (mapped.payloads[t] for t in ("flare_classify", "flare_anomaly", "get_context"))
    fact = f"{cl['model']}: {cl['predicted']} p={cl['probabilities'][cl['predicted']]}, set {cl['prediction_set']}, credibility {cl['credibility']}"
    alarm = f"{an['alarm_score']} alarm p={an['alarm_p']}, FAR={an['alarm_far']}; alarms {an['alarms']}, {an['calibration']} calibration"
    evidence = [{"tool": "flare_classify", "fact": fact}, {"tool": "flare_anomaly", "fact": alarm}]
    caveats = list(SCOPE_LIMITATIONS)
    if cx.get("host"):
        def measured(block, key):
            value = cx.get(block, {}).get(key) if isinstance(cx.get(block), dict) else None
            return "unavailable" if value is None else str(value)
        evidence.append({"tool": "get_context", "fact": f"host offset {measured('host', 'offset_arcsec')} arcsec, z_phot {measured('redshift', 'z_phot')}, M_pseudo {measured('redshift', 'M_pseudo')}, Gaia point source {measured('gaia', 'point_source_within_2arcsec')}"})
    else:
        evidence.append({"tool": "get_context", "fact": "host fact unavailable in the consulted context payload"})
    required = _or(mapped.constraints.get("class_history_trigger"), mapped.constraints.get("any_alarm"))
    if required:
        history = mapped.payloads["light_curve_history"]
        history_fact = f"full history: {history['n_detections']} detections over {history['activity_span_days']} d" if history["found"] else "completed history search found no photometry"
        evidence.append({"tool": "light_curve_history", "fact": history_fact})
    else:
        caveats.append("full photometric history not consulted (frozen legacy trigger false)")
    for tool in ("catalog_lookup", "get_photometry"):
        if tool in mapped.payloads:
            evidence.append({"tool": tool, "fact": f"decision predicate sources: { {k: v for k, v in mapped.constraints.items() if k in GROUPS[tool]} }"})
    if mapped.constraints.get("far_host"):
        caveats.append("host more than 10 arcsec away: photo-z and M_pseudo may belong to the wrong galaxy")
    if not isinstance(cx.get("redshift"), dict) or cx["redshift"].get("z_phot") is None:
        caveats.append("no host photo-z: luminosity unknown")
    raw_missing = _context_missing(cx)
    if raw_missing:
        caveats.append("missing context measurements: " + ", ".join(raw_missing))
    if not an["calibrated"]:
        caveats.append(f"anomaly alarm not calibrated at the phase of {cl['model']}: not used")
    caveats.extend(f"{tool}: {reason}" for tool, reasons in sorted(mapped.issues.items()) for reason in reasons)
    caveats.append("P(anomaly) uses the benchmark base rate 3.3%; a raw stream is poorer in rarities")
    return dict(obj_id=scope.obj_id, verdict=decision.verdict, priority=decision.priority,
                flare_label=cl["predicted"], prediction_set=cl["prediction_set"], credibility=cl["credibility"],
                energy_percentile=an["energy_percentile"], summary=f"Successful-data software policy {POLICY_ID}: {decision.verdict}. {fact}; {alarm}.",
                evidence=evidence, suggested_action=ACTIONS.get(decision.verdict, "none"), caveats=caveats)


def _context_missing(cx: Mapping[str, Any]) -> list[str]:
    fields = (("host", "offset_arcsec"), ("redshift", "z_phot"), ("redshift", "M_pseudo"),
              ("gaia", "parallax_over_error"), ("gaia", "pm_over_error"), ("wise", "w1_w2"))
    return [f"{block}.{key}" for block, key in fields if not isinstance(cx.get(block), dict) or cx[block].get(key) is None]


def _context_invalid(cx: Mapping[str, Any]) -> list[str]:
    numeric = {
        "host": ("offset_arcsec", "r_mag", "i_mag", "g_r", "extended_neighbours_30arcsec"),
        "redshift": ("z_phot", "z_spec_host", "M_pseudo"),
        "gaia": ("G", "parallax_over_error", "pm_over_error", "ruwe"),
        "wise": ("w1_w2",), "ps1_counterpart": ("r_mag", "n_detections"),
    }
    boolean = {"gaia": ("point_source_within_2arcsec",), "wise": ("match_within_6arcsec",),
               "ps1_counterpart": ("within_1p5arcsec", "point_like")}
    invalid = []
    for fields, check in ((numeric, _number), (boolean, lambda v: type(v) is bool)):
        for block, names in fields.items():
            values = cx.get(block)
            if values is not None and not isinstance(values, dict):
                invalid.append(block)
            elif isinstance(values, dict):
                invalid.extend(f"{block}.{name}" for name in names if values.get(name) is not None and not check(values[name]))
    return sorted(set(invalid))


def _flare_reachability(constraints: Mapping[str, bool], *, blind: bool = False,
                        conflicts: Sequence[str] = (), include_missing: bool = True,
                        _force_enumeration: bool = False) -> Reachability:
    """Short-circuit a sufficient singleton certificate, otherwise enumerate.

    The cardinality shortcut applies only to this adapter's full independent
    Boolean product. Generic finite-domain evaluation remains unchanged.
    """
    if set(constraints) - set(_PREDICATES) or any(type(v) is not bool for v in constraints.values()):
        raise ValueError("constraints must use declared predicate names and Boolean values")
    if not _force_enumeration:
        if conflicts:
            return Reachability(status="inconsistent", possible_decisions=(), missing_predicates=(),
                                admissible_count=0, domain_kind=_DOMAIN_KIND)
        decision = nullable_legacy_decision(constraints, blind=blind)
        if decision is not None:
            return Reachability(status="decided", possible_decisions=(decision,), missing_predicates=(),
                                admissible_count=1 << (len(_PREDICATES) - len(constraints)), domain_kind=_DOMAIN_KIND)
    return evaluate_partial(build_policy(blind), constraints, conflicts=conflicts, include_missing=include_missing)


def _evaluate_flare(scope: EvidenceScope, observations: Sequence[ToolObservation], *,
                    profile: str = "decision_only", blind: bool = False,
                    _force_enumeration: bool = False) -> PartialResult:
    if profile not in ("decision_only", "legacy_report"):
        raise ValueError("unknown report profile")
    mapped = map_observations(scope, observations, blind=blind)
    reach = _flare_reachability(mapped.constraints, blind=blind, conflicts=mapped.conflicts,
                                _force_enumeration=_force_enumeration)
    certified = reach.status == "decided"
    missing = _report_missing(mapped) if profile == "legacy_report" else []
    if not certified:
        missing.insert(0, "verdict_and_priority_certificate")
    complete = certified and not missing
    tools = tuple(t for t, names in GROUPS.items() if not (blind and t == "catalog_lookup") and
                  any(p in reach.missing_predicates for p in names))
    report = _render(scope, reach.possible_decisions[0], mapped) if complete and profile == "legacy_report" else None
    return PartialResult(scope=scope, policy_id=POLICY_ID + ("-blind" if blind else ""), profile=profile, status=reach.status,
                         possible_decisions=reach.possible_decisions, decision_certified=certified,
                         report_complete=complete, task_complete=certified and complete,
                         missing_evidence=tools, missing_report_fields=tuple(missing), used_evidence=mapped.sources,
                         rejected_evidence=mapped.issues, report=report, domain_kind=_DOMAIN_KIND,
                         metadata={"domain_notes": _DOMAIN_NOTES, "admissible_count": reach.admissible_count,
                                   "missing_evidence_is_minimal": False, "conflicts": mapped.conflicts,
                                   "scope_limitations": SCOPE_LIMITATIONS, "prose_semantically_verified": False,
                                   "observations": mapped.observations, "blind": blind,
                                   "context_raw_missing": _context_missing(mapped.payloads.get("get_context", {})),
                                   "context_invalid_fields": _context_invalid(mapped.payloads.get("get_context", {})),
                                   "context_flag_semantics": "explicit tool-computed legacy policy inputs; raw-value absence is retained, not corrected",
                                   "reference": "successful relevant scientific payloads; literal error fallbacks excluded"})


def evaluate_flare(scope: EvidenceScope, observations: Sequence[ToolObservation], *,
                   profile: str = "decision_only", blind: bool = False) -> PartialResult:
    """Evaluate captured evidence; nullable singletons avoid table construction.

    This is an engineering shortcut with the same output as finite enumeration,
    including report obligations. Unknown nullable results still enumerate.
    """
    return _evaluate_flare(scope, observations, profile=profile, blind=blind)
