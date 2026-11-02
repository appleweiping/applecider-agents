#!/usr/bin/env python3
"""Replay captured local tool observations under equal policy/report contracts.

This is policy-fidelity software evaluation, not astronomical accuracy or live
latency evaluation. No scientific backend, application LLM, or network is called.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from itertools import permutations
import json
from pathlib import Path
import time
from typing import Mapping

import numpy as np

from applecider_agents.core.acquisition import ArrayFiniteAcquisition, CostModel, StateLimitExceeded
from applecider_agents.core.evidence import ToolObservation
from applecider_agents.core.membership import load_frozen_membership
from applecider_agents.core.observations import RecordedSession, load_raw_snapshot
from applecider_agents.core.partial import Decision, evaluate_partial

ACTIONS = ('flare_classify', 'flare_anomaly', 'get_context', 'get_photometry', 'light_curve_history')
DEFAULT_ORDER = ('get_photometry', 'get_context', 'flare_classify', 'flare_anomaly', 'light_curve_history')
MASKS = {
    'clean': (), 'missing_classify': ('flare_classify',), 'missing_anomaly': ('flare_anomaly',),
    'missing_context': ('get_context',), 'missing_photometry': ('get_photometry',),
    'missing_history': ('light_curve_history',),
    'missing_context_history': ('get_context', 'light_curve_history'),
    'missing_classify_anomaly': ('flare_classify', 'flare_anomaly'),
    'wrong_object': ('flare_classify',), 'wrong_horizon': ('flare_classify',),
    'stale_session': ('flare_classify',),
}


def cost_model(kind: str) -> CostModel:
    """Predeclared imposed scenarios, separate from actual backend counters.

    Shared-cache model assumes successful reusable computation. For actual
    failures the recorded counters describe retries; this imposed model does not.
    Context and photometry share local event work; model features are a distinct
    group because captured records separate feature extraction and local event reads.
    """
    calls = dict.fromkeys(ACTIONS, 1)
    if kind == 'shared':
        return CostModel(calls, {'flare_classify': 'flare', 'flare_anomaly': 'flare',
                                'get_context': 'local', 'get_photometry': 'local',
                                'light_curve_history': 'history'},
                         {'flare': 20, 'local': 2, 'history': 8})
    if kind == 'independent':
        return CostModel(calls, {a: a for a in ACTIONS},
                         {'flare_classify': 20, 'flare_anomaly': 20,
                          'get_context': 2, 'get_photometry': 2, 'light_curve_history': 8})
    raise ValueError('unknown cost scenario')


def selected_membership(path: Path, partition: str, limit: int) -> list[str]:
    rows = load_frozen_membership(path)['rows']
    cap = {'design': 96, 'tune': 256, 'confirm': 1024}[partition]
    if not 0 < limit <= cap:
        raise ValueError(f'{partition} limit must be1..{cap}')
    return [r['obj_id'] for r in sorted((r for r in rows if r['partition'] == partition),
                                       key=lambda r: r['order_key'])[:limit]]


def load_snapshot(path, *, expected_obj=None, expected_horizon=None):
    """Accept the public typed session format and captured raw tool snapshots."""
    raw = json.loads(Path(path).read_text())
    session = RecordedSession.model_validate(raw) if 'scope' in raw else load_raw_snapshot(path)
    if expected_obj is not None and session.scope.obj_id != expected_obj:
        raise ValueError('snapshot object does not match frozen requested identity')
    if expected_horizon is not None and session.scope.horizon_days != expected_horizon:
        raise ValueError('snapshot horizon does not match frozen requested horizon')
    return session


def successful_data_reference(payloads: Mapping[str, dict], *, provenance=None, missing_history_default=False):
    """Independent literal successful-data rules; missing relevant values abstain.

    Deliberately preserve legacy detection-count-only and credibility truthiness
    rules. ``missing_history_default`` is only the literal legacy diagnostic.
    """
    cl = payloads.get('flare_classify')
    ph = payloads.get('get_photometry')
    cx = payloads.get('get_context')
    an = payloads.get('flare_anomaly')
    if cl is None or 'error' in cl:
        return Decision('insufficient_data', 0) if missing_history_default else None
    if ph is None or cx is None or an is None:
        return None
    n = ph.get('n_detections')
    if n is None or isinstance(n, bool) or not isinstance(n, (int, float)) or not np.isfinite(n) or n < 0 or int(n) != n:
        return None
    if n < 8:
        return Decision('insufficient_data', 0)
    if not missing_history_default:
        def number(value, low=None, high=None):
            return (not isinstance(value, bool) and isinstance(value, (int, float)) and
                    np.isfinite(value) and (low is None or value >= low) and (high is None or value <= high))
        classes = {'SN_Ia', 'SN_CC', 'SLSN', 'AGN', 'TDE', 'CV'}
        prediction_set = cl.get('prediction_set')
        if (not number(cl.get('credibility'), 0, 1) or not isinstance(cl.get('predicted'), str) or cl['predicted'] not in classes or
            not isinstance(prediction_set, (list, tuple)) or
            not all(isinstance(label, str) and label in classes for label in prediction_set) or
            type(cl.get('argmax_excluded_from_set')) is not bool):
            return None
        flags_check = cx.get('flags', {})
        if not isinstance(flags_check, dict) or not all(type(flags_check.get(k)) is bool for k in
            ('stellar_astrometry', 'nuclear', 'agn_mid_ir_colour', 'superluminous_if_host_is_right', 'offset_from_host_gt_10arcsec')):
            return None
        if (type(an.get('calibrated')) is not bool or 'alarm_far' not in an or
            not (an['alarm_far'] is None or number(an['alarm_far'], 0, 1)) or
            not isinstance(an.get('alarms'), dict) or type(an['alarms'].get('1%')) is not bool):
            return None
        if (not isinstance(ph.get('bands'), dict) or not number(ph.get('span_days'), 0) or
            not number(ph.get('peak_mag_observed')) or any(not isinstance(b, dict) or
                'peak_day' not in b or not (b['peak_day'] is None or number(b['peak_day']))
                for b in ph['bands'].values())):
            return None
    flags = cx.get('flags', {})
    hit = bool(an.get('calibrated', True)) and bool((an.get('alarms') or {}).get('1%', an.get('above_1pct_far')))
    trigger = (cl['credibility'] or 0) < .2 or len(cl['prediction_set']) != 1 or an.get('alarm_far') is not None
    hist = payloads.get('light_curve_history') if trigger else {}
    if flags.get('stellar_astrometry'):
        return Decision('likely_stellar_or_agn', 1 if hit else 0)
    if trigger and (flags.get('nuclear') or flags.get('agn_mid_ir_colour')) and not missing_history_default:
        if not isinstance(hist, dict):
            return None
        if hist.get('found') is False:
            source = (provenance or {}).get('light_curve_history', {})
            if source.get('completed_search') is not True:
                return None
            hist = {'activity_span_days': 0}
        elif (hist.get('found') is not True or not number(hist.get('activity_span_days'), 0) or
              not number(hist.get('n_detections'), 0) or int(hist['n_detections']) != hist['n_detections']):
            return None
    span = (hist or {}).get('activity_span_days') or 0
    if (flags.get('nuclear') or flags.get('agn_mid_ir_colour')) and span > 400:
        return Decision('likely_stellar_or_agn', 1 if hit else 0)
    young = bool(ph.get('bands')) and any(b.get('peak_day') is not None and
            ph['span_days'] - b['peak_day'] < 10 for b in ph['bands'].values())
    bright = (ph.get('peak_mag_observed') or 99) < 19
    if hit or cl['argmax_excluded_from_set'] or (cl['credibility'] or 1) < .1:
        return Decision('anomaly_review', 2 if young or flags.get('superluminous_if_host_is_right') else 1)
    if (cl['predicted'] in ('SLSN', 'TDE') or
        'TDE' in cl['prediction_set'] and flags.get('nuclear') or
        'SLSN' in cl['prediction_set'] and flags.get('superluminous_if_host_is_right') and
            not flags.get('offset_from_host_gt_10arcsec')):
        return Decision('needs_spectrum', 3 if young and bright else 2)
    return Decision('ordinary', 0)


def decode_outcome(code, width):
    """Return value bits, individually known bits and report-source validity."""
    full = (1 << width) - 1
    if isinstance(code, tuple):
        return code
    if code < 0:
        return 0, 0, False
    return int(code) & full, full, code < (1 << width)


class EC2EvidencePosterior(ArrayFiniteAcquisition):
    """Condition on every valid science bit, retaining attempted/report status.

    The finite prior's prospective tool outcomes remain unchanged. Observation
    failures reveal no science; partial payloads condition only their known bits.
    """
    def __init__(self, *args, predicate_widths, scientific_table, **kwargs):
        self.predicate_widths = predicate_widths
        self.scientific_table = scientific_table
        super().__init__(*args, **kwargs)

    def compatible(self, observed):
        key = tuple(sorted(observed.items()))
        if key not in self._compatible_cache:
            mask = self.weights > 0
            for action, code in observed.items():
                value, known, _ = decode_outcome(code, self.predicate_widths[action])
                if known:
                    column = self.scientific_table[:, self.actions.index(action)]
                    mask = mask & ((column & known) == (value & known))
            self._compatible_cache[key] = np.flatnonzero(mask)
        return self._compatible_cache[key]


class GroupedPolicyModel:
    """Dense logical support, separate scheduler weights, group outcome codes."""
    def __init__(self, profile, costs, *, prior=None, support='logical', max_states=200_000, max_seconds=30):
        from applecider_agents.flare.partial import build_policy, report_source_complete
        self.report_source_complete = report_source_complete
        self.policy = build_policy(blind=True)
        self.profile = profile
        self.groups = {a: self.policy.tool_predicates[a] for a in ACTIONS}
        self.columns = {p: j for j, p in enumerate(self.policy.predicates)}
        self.table = np.zeros((len(self.policy.states), len(ACTIONS)), dtype=np.int64)
        for j, action in enumerate(ACTIONS):
            for bit, predicate in enumerate(self.groups[action]):
                self.table[:, j] |= self.policy.states[:, self.columns[predicate]].astype(np.int64) << bit
        self.scientific_table = self.table.copy()
        # Offline history is unavailable for every replay row. Its latent policy
        # predicate still varies; no history value/probability is learned here.
        self.table[:, ACTIONS.index('light_curve_history')] = -1
        if prior is None:
            if support != 'logical':
                raise ValueError('empirical support requires a tune-only prior file')
            weights = np.ones(len(self.policy.states), dtype=float)
        else:
            raw = np.load(prior, allow_pickle=False)
            if str(raw['partition']) != 'tune' or tuple(raw['predicates']) != self.policy.predicates:
                raise ValueError('prior must be tune-only with matching predicate schema')
            empirical = raw['weights'].astype(float)
            if len(empirical) != len(self.policy.states) or empirical.sum() <= 0:
                raise ValueError('invalid tune prior weights')
            empirical /= empirical.sum()
            weights = empirical if support == 'empirical' else .95 * empirical + .05 / len(empirical)
        self.weights = weights
        decisions = sorted(set(self.policy.decisions))
        codes = {d: i for i, d in enumerate(decisions)}
        self.controls = ArrayFiniteAcquisition(ACTIONS, self.table,
            [codes[d] for d in self.policy.decisions], weights, costs,
            self.certificate, self.required, max_states=max_states, max_seconds=max_seconds)
        self.ec2_controls = EC2EvidencePosterior(ACTIONS, self.table,
            [codes[d] for d in self.policy.decisions], weights, costs,
            self.certificate, self.required, predicate_widths={a: len(self.groups[a]) for a in ACTIONS},
            scientific_table=self.scientific_table)
        self.dp_failures = {}

    def constraints(self, observed):
        constraints = {}
        for action, code in observed.items():
            value, known, _ = decode_outcome(code, len(self.groups[action]))
            constraints.update({p: bool(value & (1 << bit)) for bit, p in enumerate(self.groups[action])
                                if known & (1 << bit)})
        return constraints

    @lru_cache(maxsize=100_000)
    def _decision(self, items):
        # All logical rows participate regardless of zero/tiny scheduler weights.
        constraints = self.constraints(dict(items))
        mask = np.ones(len(self.policy.states), dtype=bool)
        for p, value in constraints.items():
            mask &= self.policy.states[:, self.columns[p]] == value
        rows = np.flatnonzero(mask)
        if len(rows) and len(np.unique(self.policy._decision_codes[rows])) == 1:
            return self.policy.decisions[int(rows[0])]
        return None

    def certificate(self, observed):
        if self._decision(tuple(sorted(observed.items()))) is None:
            return False
        return self.profile == 'decision_only' or not self.required(observed)

    def required(self, observed):
        if self.profile == 'decision_only':
            return ()
        required = [a for a in ('flare_classify', 'flare_anomaly', 'get_context')
                    if not decode_outcome(observed.get(a, -1), len(self.groups[a]))[2]]
        constraints = self.constraints(observed)
        if constraints.get('class_history_trigger') or constraints.get('any_alarm'):
            if not decode_outcome(observed.get('light_curve_history', -1), len(self.groups['light_curve_history']))[2]:
                required.append('light_curve_history')
        return tuple(required)

    def outcome(self, action, mapping):
        width = len(self.groups[action])
        value = sum(int(mapping.constraints[p]) << bit for bit, p in enumerate(self.groups[action])
                    if p in mapping.constraints)
        known = sum(1 << bit for bit, p in enumerate(self.groups[action]) if p in mapping.constraints)
        report_valid = action in mapping.usable_tools and (self.profile == 'decision_only' or self.report_source_complete(mapping, action))
        if known == 0 and (self.profile == 'decision_only' or not report_valid):
            return -1
        if known == (1 << width) - 1:
            return value if report_valid else value | (1 << width)
        return (value, known, report_valid) if known or report_valid else -1


def intervention(session, name):
    """Mask existing valid payloads only; absent history is never invented."""
    from applecider_agents.flare.partial import map_observations
    requested = MASKS[name]
    mapped = map_observations(session.scope, session.observations, blind=True)
    eligible = tuple(a for a in requested if a in mapped.valid_tools)
    rows = []
    for observation in session.observations:
        if observation.tool in eligible and name in ('wrong_object', 'wrong_horizon', 'stale_session'):
            scope = observation.scope
            if name == 'wrong_object':
                scope = scope.model_copy(update={'obj_id': scope.obj_id + ':controlled-wrong-object'})
            elif name == 'wrong_horizon':
                scope = scope.model_copy(update={'horizon_days': scope.horizon_days + 1.0})
            else:
                scope = scope.model_copy(update={'session_id': scope.session_id + ':controlled-stale-session'})
            observation = observation.model_copy(update={'scope': scope})
        elif observation.tool in eligible:
            observation = observation.model_copy(update={
                'status': 'unavailable', 'payload': None,
                'availability': {'reason': 'controlled_post_acquisition_evidence_loss',
                                 'original_payload_preserved_in_source_snapshot': True}})
        rows.append(observation)
    return tuple(rows), eligible


def recorded_work(session):
    """Retain actual collector/imported acquisition receipts without zero filling.

    Imported raw captures nest per-tool receipts under recorded_acquisition;
    native typed collection keeps per-tool deltas directly in provenance and
    aggregate backend records in session.costs. Their timing scopes stay named.
    """
    totals = defaultdict(int)
    nested_records = []
    observation_walls = []
    deltas_recorded = False
    backend_records_recorded = False
    for observation in session.observations:
        provenance = observation.provenance
        raw = provenance.get('recorded_acquisition', provenance)
        if isinstance(raw.get('wall_seconds'), (int, float)):
            observation_walls.append(raw['wall_seconds'])
        delta = raw.get('backend_work_delta')
        if isinstance(delta, dict):
            deltas_recorded = True
            for key, value in delta.items():
                totals[key] += value
        if isinstance(raw.get('backend_records'), list):
            backend_records_recorded = True
            nested_records.extend(raw['backend_records'])
    costs = session.costs
    if 'wall_seconds' in costs:
        wall = costs['wall_seconds']
        scope = 'native typed collector whole-session wall'
    elif 'recorded_session_wall_seconds' in costs:
        wall = costs['recorded_session_wall_seconds']
        scope = 'imported raw collector whole-session wall'
    else:
        wall, scope = None, 'whole-session wall unavailable'
    if 'backend_records' in costs:
        records = costs['backend_records']
    else:
        records = nested_records if backend_records_recorded else None
    return {'source_acquisition_wall_seconds': wall, 'source_acquisition_wall_scope': scope,
            'source_observation_wall_seconds': sum(observation_walls) if observation_walls else None,
            'source_backend_work_delta': dict(totals) if deltas_recorded else None,
            'source_backend_records': records, 'source_session_costs': dict(costs),
            'interpretation': 'actual source acquisition receipt, not reordered live acquisition savings'}


def run_case(session, model, *, method, budget, order=DEFAULT_ORDER, scenario='clean', missing_semantics='unknown'):
    from applecider_agents.flare.partial import evaluate_flare, map_observations, nullable_legacy_decision
    started = time.perf_counter()
    observations, eligible = intervention(session, scenario)
    by_tool = {o.tool: o for o in observations}
    original = map_observations(session.scope, session.observations, blind=True)
    reference = successful_data_reference(original.payloads,
        provenance={tool: record.get('provenance', {}) for tool, record in original.observations.items()})
    acquired = []
    observed = {}
    failure = None
    spent = 0
    dp_values = []
    while True:
        current = [by_tool[a] for a in acquired if a in by_tool]
        actual = evaluate_flare(session.scope, current, profile=model.profile, blind=True)
        mapped = map_observations(session.scope, current, blind=True)
        direct = nullable_legacy_decision(mapped.constraints, blind=True)
        complete = actual.task_complete and (method != 'fail_closed' or direct is not None)
        if complete and method not in ('eager', 'legacy_eager_prefix'):
            break
        remaining = budget - spent
        try:
            if method in ('eager', 'legacy_eager_prefix'):
                trigger = mapped.constraints.get('class_history_trigger') or mapped.constraints.get('any_alarm')
                history_needed = trigger or not all(p in mapped.constraints for p in ('class_history_trigger', 'any_alarm'))
                eager_order = ACTIONS if history_needed else ACTIONS[:-1]
                action = next((a for a in eager_order if a not in acquired and
                    model.controls.costs.increment(a, acquired) <= remaining), None)
            elif method == 'compiled':
                action = model.controls.fixed_next(observed, remaining, order)
            elif method == 'fail_closed':
                # Direct nullable execution uses a declared syntactic order,
                # abstaining immediately when a needed condition is unavailable.
                # Before direct resolution, acquire the next needed tool in
                # fixed legacy branch order; failed required tool closes flow.
                needed = direct_needed_tool(mapped.constraints)
                if needed is not None and needed in acquired:
                    action = None
                elif direct is None:
                    action = needed
                else:
                    action = next((a for a in model.required(observed) if a not in acquired), None)
                if action and model.controls.costs.increment(action, acquired) > remaining:
                    action = None
            elif method == 'ec2':
                action = model.ec2_controls.ec2_next(observed, remaining)
            elif method == 'exact_dp':
                if scenario != 'clean':
                    raise ValueError('exact DP replay unavailable: controlled evidence-loss/scope faults are outside declared finite outcome prior')
                dp_key = (tuple(sorted(observed.items())), remaining)
                if dp_key in model.dp_failures:
                    raise StateLimitExceeded(model.dp_failures[dp_key])
                try:
                    value = model.controls.exact_dp(observed, remaining)
                except StateLimitExceeded as exc:
                    model.dp_failures[dp_key] = str(exc)
                    raise
                dp_values.append({'success_probability': value.success_probability,
                                  'expected_spent_cost': value.expected_spent_cost,
                                  'states': model.controls.dp_states})
                action = value.action
            else:
                raise ValueError(f'unknown method {method}')
        except (StateLimitExceeded, ValueError) as exc:
            failure = f'{type(exc).__name__}: {exc}'
            break
        if action is None:
            break
        spent += model.controls.costs.increment(action, acquired)
        acquired.append(action)
        if action not in by_tool:
            observed[action] = -1
        else:
            one = map_observations(session.scope, [by_tool[action]], blind=True)
            observed[action] = model.outcome(action, one)
        if missing_semantics == 'negative':
            # Deliberately treat unknown bits as false and missing report sources
            # as supplied, then audit against actual evidence below.
            value, _, _ = decode_outcome(observed[action], len(model.groups[action]))
            observed[action] = value
    current = [by_tool[a] for a in acquired if a in by_tool]
    actual = evaluate_flare(session.scope, current, profile=model.profile, blind=True)
    mapped = map_observations(session.scope, current, blind=True)
    direct = nullable_legacy_decision(mapped.constraints, blind=True)
    decision = actual.decision
    if method == 'fail_closed' and direct is None:
        decision = None
    if missing_semantics == 'negative':
        decision = model._decision(tuple(sorted(observed.items())))
    if method == 'legacy_eager_prefix':
        decision = successful_data_reference(mapped.payloads, missing_history_default=True)
    internally_supported = decision is not None and actual.decision_certified and decision == actual.decision
    reference_mismatch = decision != reference if decision is not None and reference is not None else None
    supported = internally_supported and reference_mismatch is not True
    completed = supported and (model.profile == 'decision_only' or actual.report_complete)
    return {
        'obj_id': session.scope.obj_id, 'horizon_days': session.scope.horizon_days,
        'profile': model.profile, 'method': method, 'budget': budget, 'scenario': scenario,
        'intervention_kind': 'none' if scenario == 'clean' else ('controlled_scope_token_mismatch' if scenario in ('wrong_object', 'wrong_horizon', 'stale_session') else 'controlled_post_acquisition_evidence_loss'),
        'scope_fault_transform': {'wrong_object': 'append :controlled-wrong-object to classify scope obj_id',
            'wrong_horizon': 'add1 day to classify captured scope horizon only',
            'stale_session': 'append :controlled-stale-session to classify scope session_id; not an actual cache-reseed failure'}.get(scenario),
        'requested_mask': list(MASKS[scenario]), 'eligible_mask': list(eligible),
        'intervention_eligible': bool(eligible) if scenario != 'clean' else True,
        'missing_semantics': missing_semantics, 'decision': decision, 'decision_certified': supported,
        'engine_certificate_valid': internally_supported, 'internally_certified': internally_supported,
        'independent_reference_available': reference is not None,
        'independent_reference_error': reference_mismatch,
        'independently_verified_decision': internally_supported and reference is not None and reference_mismatch is False,
        'emitted_singleton_without_certificate': decision is not None and not internally_supported,
        'independent_reference_mismatch': reference_mismatch,
        'report_complete': actual.report_complete, 'task_complete': completed,
        'unsupported_singleton': decision is not None and not supported,
        'possible_decisions': actual.possible_decisions, 'candidate_set_size': len(actual.possible_decisions),
        'reference': reference, 'reference_kind': 'successful_data_policy' if reference else 'no_reference',
        'reference_agreement': decision == reference if decision is not None and reference is not None else None,
        'logical_tool_calls': len(acquired), 'acquisition_order': acquired, 'imposed_cost_units': spent,
        'imposed_unique_backend_groups': model.controls.costs.unique_backends(acquired),
        'replay_controller_wall_seconds': time.perf_counter() - started,
        'missing_report_fields': actual.missing_report_fields, 'failure': failure,
        'dp_values': dp_values, 'source_recorded_costs': recorded_work(session),
    }


def direct_needed_tool(c):
    """Next nullable policy dependency, respecting Python short-circuit order."""
    def missing(names):
        return any(p not in c for p in names)
    if missing(('enough_detections',)):
        return 'get_photometry'
    if not c['enough_detections']:
        return None
    if missing(('stellar',)):
        return 'get_context'
    if c['stellar']:
        return 'flare_anomaly' if 'alarm_hit' not in c else None
    if missing(('nuclear', 'agn_colour')):
        return 'get_context'
    if c['nuclear'] or c['agn_colour']:
        if 'class_history_trigger' not in c:
            return 'flare_classify'
        if not c['class_history_trigger'] and 'any_alarm' not in c:
            return 'flare_anomaly'
        if (c['class_history_trigger'] or c.get('any_alarm')) and 'long_history' not in c:
            return 'light_curve_history'
        if c.get('long_history') and (c['class_history_trigger'] or c.get('any_alarm')):
            return 'flare_anomaly' if 'alarm_hit' not in c else None
    if 'alarm_hit' not in c:
        return 'flare_anomaly'
    if not c['alarm_hit'] and 'class_anomaly' not in c:
        return 'flare_classify'
    if c['alarm_hit'] or c.get('class_anomaly'):
        if 'young' not in c:
            return 'get_photometry'
        return 'get_context' if not c['young'] and 'luminous' not in c else None
    if missing(('rare_prediction', 'set_tde', 'set_slsn')):
        return 'flare_classify'
    if missing(('luminous', 'far_host')):
        return 'get_context'
    rare = c['rare_prediction'] or c['set_tde'] and c['nuclear'] or c['set_slsn'] and c['luminous'] and not c['far_host']
    if rare and missing(('young', 'bright')):
        return 'get_photometry'
    return None


def failed_request_row(obj, horizon, profile, method, budget, scenario, missing_semantics, error):
    return dict(obj_id=obj, horizon_days=horizon, profile=profile, method=method,
        budget=budget, scenario=scenario, missing_semantics=missing_semantics,
        decision=None, reference=None, decision_certified=False, report_complete=False,
        task_complete=False, unsupported_singleton=False, reference_kind='no_reference',
        reference_agreement=None, imposed_cost_units=None, logical_tool_calls=None,
        cost_accounting='unknown_after_request_execution_failure', failure=error, missing_report_fields=['request_execution_failed'])


def summarize(rows):
    groups = defaultdict(list)
    keys = ('method', 'profile', 'budget', 'scenario', 'cost_scenario', 'support', 'missing_semantics')
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    output = []
    for key, values in groups.items():
        n = len(values)
        errors = sum(r['unsupported_singleton'] for r in values)
        reference_looks = [r for r in values if r['reference_kind'] != 'no_reference']
        comparisons = [r for r in values if r['reference_agreement'] is not None]
        compared_objects = {r['obj_id'] for r in comparisons}
        compared_error_objects = {r['obj_id'] for r in comparisons if r['reference_agreement'] is False}
        object_ids = {r['obj_id'] for r in values}
        error_objects = {r['obj_id'] for r in values if r['unsupported_singleton']}
        output.append({**dict(zip(keys, key)), 'n_requested': n,
            'n_objects': len({r['obj_id'] for r in values}),
            'decision_certified': sum(r['decision_certified'] for r in values),
            'tasks_complete': sum(r['task_complete'] for r in values),
            'reports_complete': sum(r['report_complete'] for r in values),
            'unsupported_singletons': errors,
            'snapshot_error_count_descriptive': f'{errors}/{n}',
            'objects_with_any_unsupported_singleton': len(error_objects),
            'zero_error_object_two_sided_95_upper': 1 - .025 ** (1 / len(object_ids)) if not error_objects else None,
            'task_completion_rate': np.mean([r['task_complete'] for r in values]),
            'requests_with_known_spent_cost': sum(r['imposed_cost_units'] is not None for r in values),
            'mean_spent_cost_all_requests': float(np.mean([r['imposed_cost_units'] for r in values])) if all(r['imposed_cost_units'] is not None for r in values) else None,
            'mean_logical_calls_all_requests': float(np.mean([r['logical_tool_calls'] for r in values])) if all(r['logical_tool_calls'] is not None for r in values) else None,
            'failures': sum(r['failure'] is not None for r in values),
            'reference_available': len(reference_looks),
            'reference_eligible_objects': len({r['obj_id'] for r in reference_looks}),
            'no_reference_looks': n - len(reference_looks),
            'independently_compared_objects': len(compared_objects),
            'independent_error_objects': len(compared_error_objects),
            'independent_zero_error_object_two_sided_95_upper': 1 - .025 ** (1 / len(compared_objects)) if compared_objects and not compared_error_objects else None,
            'independent_bound_scope': 'only emitted referenced software-policy looks; undefined and unemitted looks excluded from this empirical risk claim',
            'reference_comparisons': sum(r['reference_agreement'] is not None for r in values),
            'reference_disagreements': sum(r['reference_agreement'] is False for r in values)})
    return output


def paired_bootstrap(rows, baseline='compiled', replicates=2000, seed=20261102):
    """Object-clustered paired differences, retaining every requested input."""
    cells = defaultdict(dict)
    for row in rows:
        key = (row['profile'], row['budget'], row['scenario'], row['cost_scenario'],
               row['support'], row['missing_semantics'], row['obj_id'], row['horizon_days'])
        cells[key][row['method']] = row
    contrasts = defaultdict(list)
    for key, methods in cells.items():
        if baseline not in methods:
            continue
        base = methods[baseline]
        for name, row in methods.items():
            if name != baseline:
                contrasts[(*key[:6], name)].append((key[6], row, base))
    rng = np.random.default_rng(seed)
    results = []
    for key, values in contrasts.items():
        objects = sorted({obj for obj, _, _ in values})
        known_costs = all(r['imposed_cost_units'] is not None and b['imposed_cost_units'] is not None for _, r, b in values)
        data = []
        for obj in objects:
            pairs = [(r, b) for name, r, b in values if name == obj]
            data.append([np.mean([r['imposed_cost_units'] - b['imposed_cost_units'] for r, b in pairs]) if known_costs else 0,
                         np.mean([int(r['task_complete']) - int(b['task_complete']) for r, b in pairs])])
        data = np.asarray(data)
        boot = data[rng.integers(0, len(data), size=(replicates, len(data)))].mean(axis=1)
        both = [(r, b) for _, r, b in values if r['task_complete'] and b['task_complete'] and r['imposed_cost_units'] is not None and b['imposed_cost_units'] is not None]
        results.append({'profile': key[0], 'budget': key[1], 'scenario': key[2],
            'cost_scenario': key[3], 'support': key[4], 'missing_semantics': key[5],
            'method': key[6], 'baseline': baseline, 'n_pairs': len(values), 'n_objects': len(objects),
            'cost_difference_all_requests': float(data[:, 0].mean()) if known_costs else None,
            'cost_difference_all_requests_ci95': np.quantile(boot[:, 0], [.025, .975]).tolist() if known_costs else None,
            'completion_rate_difference': float(data[:, 1].mean()),
            'completion_rate_difference_ci95': np.quantile(boot[:, 1], [.025, .975]).tolist(),
            'n_both_complete': len(both),
            'cost_difference_resolvable_paired_subset': float(np.mean([r['imposed_cost_units'] - b['imposed_cost_units'] for r, b in both])) if both else None,
            'replicates': replicates, 'seed': seed})
    return results


def score_fixed_order(completion, costs, budget, order):
    """Simulate a fixed order using actual per-subset runtime completion flags."""
    subset, spent, acquired = 0, 0, []
    for action in order:
        if completion[subset]:
            break
        cost = costs.increment(action, acquired)
        if cost > budget - spent:
            continue
        acquired.append(action)
        subset |= 1 << ACTIONS.index(action)
        spent += cost
    return bool(completion[subset]), spent


def fit_tune_configuration(sessions, output, *, budgets, cost_scenarios, profiles):
    """Choose fixed orders and prior weights exclusively from frozen tune rows.

    The unavailable latent history predicate receives uniform modeling mass
    within each compatible row set. It is never treated as a measured outcome.
    Fit records use only policy predicates, never spectral truth or npz labels.
    """
    from applecider_agents.flare.partial import map_observations, evaluate_flare
    models = {profile: GroupedPolicyModel(profile, cost_model('shared')) for profile in profiles}
    policy = next(iter(models.values())).policy
    weights = np.zeros(len(policy.states), dtype=float)
    for session in sessions:
        mapped = map_observations(session.scope, session.observations, blind=True)
        mask = np.ones(len(policy.states), dtype=bool)
        for predicate, value in mapped.constraints.items():
            mask &= policy.states[:, policy.predicates.index(predicate)] == value
        if mask.any():
            weights[mask] += 1 / mask.sum()
    output.parent.mkdir(parents=True, exist_ok=True)
    prior_path = output.with_suffix('.prior.npz')
    np.savez_compressed(prior_path, weights=weights, partition=np.array('tune'),
                        predicates=np.array(policy.predicates),
                        object_ids=np.array(sorted({s.scope.obj_id for s in sessions})))
    orders = {}
    scores = {}
    for profile, model in models.items():
        completion_tables = []
        for session in sessions:
            table = []
            for subset in range(1 << len(ACTIONS)):
                selected = {a for j, a in enumerate(ACTIONS) if subset & (1 << j)}
                table.append(evaluate_flare(session.scope, [o for o in session.observations if o.tool in selected],
                    profile=profile, blind=True).task_complete)
            completion_tables.append(table)
        for scenario in cost_scenarios:
            costs = cost_model(scenario)
            for budget in budgets:
                best_key = None
                best_order = None
                for order in permutations(ACTIONS):
                    successes, total_spent = 0, 0
                    for completion in completion_tables:
                        completed, spent = score_fixed_order(completion, costs, budget, order)
                        successes += completed
                        total_spent += spent
                    key = (-successes, total_spent, order)
                    if best_key is None or key < best_key:
                        best_key, best_order = key, order
                name = f'{profile}|{scenario}|{budget}'
                orders[name] = best_order
                scores[name] = {'n_complete': -best_key[0], 'total_spent': best_key[1],
                                'n_requested': len(completion_tables), 'permutations_evaluated': 120}
    config = {'partition': 'tune', 'n_sessions': len(sessions),
        'n_objects': len({s.scope.obj_id for s in sessions}), 'compiled_orders': orders,
        'tune_order_scores': scores, 'prior_path': str(prior_path),
        'history_prior': 'uniform latent modeling states, no measured history outcomes',
        'objective': 'maximize actual same-profile runtime subset completion then minimize all-request imposed spend',
        'tune_terminal_model': 'actual evaluate_flare task_complete for all32 acquired-tool subsets per session/profile',
        'logical_smoothing_mass': .05, 'bootstrap_replicates': 2000, 'bootstrap_seed': 20261102}
    output.write_text(json.dumps(config, indent=2) + '\n')
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--membership', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partition', choices=('design', 'tune', 'confirm'), default='design')
    parser.add_argument('--limit', type=int, default=96)
    parser.add_argument('--methods', default='eager,compiled,fail_closed,ec2')
    parser.add_argument('--profiles', default='decision_only,legacy_report')
    parser.add_argument('--budgets', default='40')
    parser.add_argument('--scenarios', default='clean')
    parser.add_argument('--cost-scenarios', default='shared')
    parser.add_argument('--support', choices=('logical', 'empirical'), default='logical')
    parser.add_argument('--prior', type=Path)
    parser.add_argument('--frozen-config', type=Path)
    parser.add_argument('--tune-config-out', type=Path)
    parser.add_argument('--missing-semantics', choices=('unknown', 'negative'), default='unknown')
    parser.add_argument('--max-dp-states', type=int, default=200_000)
    parser.add_argument('--max-dp-seconds', type=float, default=30)
    args = parser.parse_args(argv)
    if args.partition == 'confirm' and args.frozen_config is None:
        parser.error('confirmation replay requires a frozen configuration')
    config = json.loads(args.frozen_config.read_text()) if args.frozen_config else {}
    order = tuple(config.get('compiled_order', DEFAULT_ORDER))
    if args.prior is None and config.get('prior_path'):
        args.prior = Path(config['prior_path'])
    membership = load_frozen_membership(args.membership)
    ids = selected_membership(args.membership, args.partition, args.limit)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'rows.jsonl').exists():
        parser.error('output rows already exist; choose a fresh output directory')
    rows = []
    started = time.time()
    repo = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__).resolve(), *[repo / 'applecider_agents' / p for p in
        ('core/acquisition.py', 'core/evidence.py', 'core/partial.py', 'core/observations.py', 'core/membership.py', 'flare/partial.py')]]
    source_start = {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    methods = args.methods.split(',')
    scenarios = args.scenarios.split(',')
    unknown = set(scenarios) - set(MASKS)
    if unknown:
        parser.error(f'unknown scenarios: {unknown}')
    sessions = []
    load_failures = []
    input_hashes = {}
    for obj in ids:
        for horizon in (5, 10, 20, 100):
            path = args.snapshots / f'{obj}_h{horizon}.json'
            try:
                sessions.append(load_snapshot(path, expected_obj=obj, expected_horizon=horizon))
                input_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception as exc:
                load_failures.append((obj, horizon, f'{type(exc).__name__}: {exc}'))
    if args.tune_config_out:
        if args.partition != 'tune' or args.limit != 256 or load_failures:
            parser.error('tune selection requires all first256 tune objects with every snapshot available')
        config = fit_tune_configuration(sessions, args.tune_config_out, budgets=list(map(int, args.budgets.split(','))),
            cost_scenarios=args.cost_scenarios.split(','), profiles=args.profiles.split(','))
        args.prior = Path(config['prior_path'])
    with (args.output / 'rows.jsonl').open('w') as stream:
        for costs in args.cost_scenarios.split(','):
            for profile in args.profiles.split(','):
                model = GroupedPolicyModel(profile, cost_model(costs), prior=args.prior,
                                           support=args.support, max_states=args.max_dp_states, max_seconds=args.max_dp_seconds)
                for budget in map(int, args.budgets.split(',')):
                    order = tuple(config.get('compiled_orders', {}).get(f'{profile}|{costs}|{budget}', order))
                    for scenario in scenarios:
                        for session in sessions:
                            for method in methods:
                                try:
                                    row = run_case(session, model, method=method, budget=budget, order=order,
                                                   scenario=scenario, missing_semantics=args.missing_semantics)
                                except Exception as exc:
                                    row = failed_request_row(session.scope.obj_id, session.scope.horizon_days,
                                        profile, method, budget, scenario, args.missing_semantics,
                                        f'{type(exc).__name__}: {exc}')
                                row.update(cost_scenario=costs, support=args.support)
                                rows.append(row)
                                stream.write(json.dumps(row, allow_nan=False) + '\n')
                                stream.flush()
                        for obj, horizon, error in load_failures:
                            for method in methods:
                                row = dict(obj_id=obj, horizon_days=horizon, profile=profile, method=method,
                                    budget=budget, scenario=scenario, cost_scenario=costs, support=args.support,
                                    missing_semantics=args.missing_semantics, decision=None, reference=None,
                                    decision_certified=False, report_complete=False, task_complete=False,
                                    unsupported_singleton=False, reference_kind='no_reference', reference_agreement=None,
                                    imposed_cost_units=0, logical_tool_calls=0, failure=error,
                                    missing_report_fields=['snapshot_unavailable'])
                                rows.append(row)
                                stream.write(json.dumps(row, allow_nan=False) + '\n')
    summary = summarize(rows)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (args.output / 'paired_bootstrap.json').write_text(json.dumps(paired_bootstrap(rows), indent=2, allow_nan=False) + '\n')
    manifest = {'partition': args.partition, 'n_objects_requested': len(ids), 'n_sessions_requested': len(ids) * 4,
        'n_rows': len(rows), 'load_failures': len(load_failures),
        'source_sha256_start': source_start,
        'membership_raw_sha256': membership['raw_sha256'],
        'membership_semantic_row_sha256': membership['semantic_row_sha256'],
        'input_snapshot_sha256': input_hashes,
        'source_sha256_end': {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}, 'elapsed_wall_seconds': time.time() - started,
        'compiled_order': order, 'order_status': 'tune_frozen' if args.frozen_config else ('tune_selected' if args.tune_config_out else 'provisional_design_default'),
        'logical_support': 'conservative_boolean_overapproximation', 'prior': str(args.prior) if args.prior else 'declared_uniform_logical_model',
        'history_probability': 'not_empirically_estimated; latent Boolean modeling states; observed offline history unavailable',
        'exact_dp_scope': 'supplied successful-scientific-payload finite prior with offline unavailable history; bounded integer budget; malformed report fields/nonhistory masks are outside outcome model, failures retained',
        'costs': 'imposed shared-success-cache/independent scenarios; recorded source work separate; no live latency savings',
        'evaluation_type': 'software_policy_fidelity_not_astronomical_ground_truth',
        'error_scope': 'unsupported emission or available independent software-reference mismatch; no_reference singleton is not empirically verified',
        'classifier_overlap': 'train/validation arrays overlap published classifier training/calibration',
        'frozen_configuration_sha256': hashlib.sha256(args.frozen_config.read_bytes()).hexdigest() if args.frozen_config else None,
        'scheduler_prior_file_sha256': hashlib.sha256(args.prior.read_bytes()).hexdigest() if args.prior else None,
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    manifest['source_unchanged_during_replay'] = manifest['source_sha256_start'] == manifest['source_sha256_end']
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'rows': len(rows), 'summary': str(args.output / 'summary.json'), 'wall_seconds': manifest['elapsed_wall_seconds']}))


if __name__ == '__main__':
    main()
