#!/usr/bin/env python3
"""Replay the actual pinned legacy function on matched captured tool payloads.

This diagnostic is not a live legacy-pipeline replay or a budgeted scheduler
comparison. In particular get_context receives the captured request-prefix
payload, even though the original live call omitted its horizon argument.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from applecider_agents.core.report import Report, Evidence
from run_partial_benchmark import load_snapshot, selected_membership, intervention, MASKS

BASE = '789b7955dc1ede4a7d7ec1a74d8d571e3528eba9'
TRIAGE_PATH = 'applecider_agents/flare/triage.py'


def pinned_function(repo):
    """Use the unchanged source function, with a private tool namespace per run."""
    source = subprocess.check_output(['git', 'show', f'{BASE}:{TRIAGE_PATH}'], cwd=repo)
    current = (repo / TRIAGE_PATH).read_bytes()
    if hashlib.sha256(source).digest() != hashlib.sha256(current).digest():
        raise ValueError('current triage.py differs from pinned legacy source')
    tree = ast.parse(source.decode())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'mock_triage')
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    return compile(module, '<pinned-legacy-diagnostic>', 'exec'), hashlib.sha256(source).hexdigest()


def replay(session, code):
    calls = []
    availability_adaptations = []
    by_tool = {o.tool: o for o in session.observations}
    def tool(name):
        def call(arguments):
            calls.append({'tool': name, 'arguments': deepcopy(arguments)})
            observation = by_tool.get(name)
            if observation is not None and observation.status == 'success' and observation.payload is not None:
                return deepcopy(observation.payload)
            if observation is not None and observation.payload is not None and 'error' in observation.payload:
                return deepcopy(observation.payload)
            status = observation.status if observation is not None else 'unattempted'
            reason = observation.availability.get('reason') if observation is not None else 'no captured observation'
            envelope = {'error': f'{status}: {reason or "no successful scientific payload"}'}
            availability_adaptations.append({'tool': name, 'captured_status': status,
                'replay_error_envelope': envelope, 'is_measured_scientific_value': False})
            return envelope
        return SimpleNamespace(call=call)
    namespace = {'Report': Report, 'Evidence': Evidence,
        'O': SimpleNamespace(**{n: tool(n) for n in ('catalog_lookup', 'get_context', 'get_photometry', 'light_curve_history')}),
        'T': SimpleNamespace(**{n: tool(n) for n in ('flare_classify', 'flare_anomaly')})}
    exec(code, namespace)
    started = time.perf_counter()
    report, failure = None, None
    try:
        report = namespace['mock_triage'](session.scope.obj_id, session.scope.horizon_days, blind=True)
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
    return {'obj_id': session.scope.obj_id, 'horizon_days': session.scope.horizon_days,
        'policy_id': 'literal-legacy-v1', 'diagnostic_only': True,
        'payload_scope': 'matched captured prefix; not live default100 context pipeline',
        'literal_report': report.model_dump(mode='json') if report is not None else None,
        'failure': failure, 'logical_calls': calls, 'n_logical_calls': len(calls),
        'availability_adaptations': availability_adaptations,
        'history_reference_invented': False, 'new_profile_certified': False,
        'controller_wall_seconds': time.perf_counter() - started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--membership', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenarios', default='clean')
    parser.add_argument('--partition', choices=('design', 'tune', 'confirm'), default='design')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(argv)
    if args.limit is None:
        args.limit = {'design': 96, 'tune': 256, 'confirm': 1024}[args.partition]
    scenarios = args.scenarios.split(',')
    if set(scenarios) - set(MASKS) or len(set(scenarios)) != len(scenarios):
        parser.error('scenarios must be unique declared intervention names')
    repo = Path(__file__).resolve().parents[1]
    code, source_hash = pinned_function(repo)
    ids = selected_membership(args.membership, args.partition, args.limit)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'rows.jsonl').exists():
        parser.error('choose a fresh output directory')
    rows = []
    inputs = {}
    for obj in ids:
        for horizon in (5, 10, 20, 100):
            path = args.snapshots / f'{obj}_h{horizon}.json'
            try:
                session = load_snapshot(path, expected_obj=obj, expected_horizon=horizon)
                inputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                load_failure = None
            except Exception as exc:
                session = None
                load_failure = f'{type(exc).__name__}: {exc}'
            for scenario in scenarios:
                if session is None:
                    row = {'obj_id': obj, 'horizon_days': horizon, 'policy_id': 'literal-legacy-v1',
                        'diagnostic_only': True, 'literal_report': None, 'failure': load_failure,
                        'logical_calls': [], 'new_profile_certified': False}
                else:
                    observations, eligible = intervention(session, scenario)
                    row = replay(session.model_copy(update={'observations': observations}), code)
                    row.update(eligible_intervention_tools=list(eligible), requested_intervention_tools=list(MASKS[scenario]))
                row['scenario'] = scenario
                rows.append(row)
    (args.output / 'rows.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
    manifest = {'base_revision': BASE, 'triage_sha256': source_hash, 'current_triage_matches_base': True,
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'matched_loader_sha256': hashlib.sha256((repo / 'scripts/run_partial_benchmark.py').read_bytes()).hexdigest(),
        'partition': args.partition, 'n_requested_objects': len(ids), 'n_requested_looks': len(ids) * 4, 'n_requested_rows': len(ids) * 4 * len(scenarios),
        'scenarios': scenarios,
        'n_rows': len(rows), 'failures': sum(r['failure'] is not None for r in rows),
        'input_sha256': inputs, 'membership_sha256': hashlib.sha256(args.membership.read_bytes()).hexdigest(),
        'scope': 'actual pinned function, matched captured payload replay, blind catalogue configuration',
        'timing': 'controller replay only, no model/network calls',
        'unavailable': 'known status adapted to explicit error envelope, never a measured negative or history reference',
        'fair_cost_comparison_eligible': False, 'new_profile_certified': False,
        'no_live_default100_pipeline_claim': True, 'no_scientific_accuracy_claim': True}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'rows': len(rows), 'failures': manifest['failures'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
