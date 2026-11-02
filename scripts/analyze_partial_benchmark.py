#!/usr/bin/env python3
"""Validate complete replay grids and compute explicit paired scientific summaries.

Raw rows may remain external. No science/model/network tools are called. Missing
cost accounting suppresses a cost contrast; failed/unresolved inputs stay paired.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np

from run_partial_benchmark import paired_bootstrap, selected_membership

CELL = ('profile', 'budget', 'scenario', 'cost_scenario', 'support', 'missing_semantics', 'method')
LOOK = ('obj_id', 'horizon_days')
FLAGS = ('decision_certified', 'report_complete', 'task_complete', 'unsupported_singleton')
KEEP = set(CELL + LOOK + FLAGS + ('failure', 'reference_kind', 'reference_agreement',
    'imposed_cost_units', 'logical_tool_calls', 'eligible_mask', 'requested_mask', 'acquisition_order'))


def read_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open() as stream:
            for line in stream:
                row = json.loads(line)
                required = set(CELL + LOOK + FLAGS + ('failure', 'reference_kind', 'reference_agreement', 'imposed_cost_units'))
                if not required <= row.keys():
                    raise ValueError('row lacks required identity/status/reference/cost fields')
                if any(type(row[name]) is not bool for name in FLAGS):
                    raise ValueError('completion/error flags must be booleans')
                rows.append({name: value for name, value in row.items() if name in KEEP})
    return rows


def validate_cartesian(rows, *, objects, profiles, budgets, costs, methods,
                       supports=('logical',), scenarios=('clean',), missing_semantics=('unknown',),
                       horizons=(5, 10, 20, 100)):
    if len(set(objects)) != len(objects) or not objects:
        raise ValueError('expected objects must be nonempty and unique')
    cells = set(product(profiles, budgets, scenarios, costs, supports, missing_semantics, methods))
    looks = set(product(objects, map(float, horizons)))
    seen = set()
    for row in rows:
        look = (row['obj_id'], float(row['horizon_days']))
        cell = tuple(row[k] for k in CELL)
        if look not in looks or cell not in cells:
            raise ValueError(f'row outside declared Cartesian grid: {look + cell}')
        key = look + cell
        if key in seen:
            raise ValueError(f'duplicate requested grid row: {key}')
        seen.add(key)
    expected = len(looks) * len(cells)
    if len(seen) != expected:
        missing = next((look + cell for look in looks for cell in cells if look + cell not in seen), None)
        raise ValueError(f'incomplete requested grid: expected{expected}, got{len(seen)}, example{missing}')
    return {'planned_rows': expected, 'completed_rows': len(seen), 'n_objects': len(objects),
            'n_horizons': len(tuple(horizons)), 'n_cells': len(cells), 'unique_complete_grid': True}


def aggregate_cell(rows):
    objects = {r['obj_id'] for r in rows}
    compared = [r for r in rows if r['reference_agreement'] is not None]
    compared_objects = {r['obj_id'] for r in compared}
    error_objects = {r['obj_id'] for r in compared if r['reference_agreement'] is False}
    complete = [r for r in rows if r['task_complete']]
    costs = [r['imposed_cost_units'] for r in rows]
    known = all(c is not None for c in costs)
    eligible = [r for r in rows if r.get('eligible_mask')]
    return {'n_requested': len(rows), 'n_objects': len(objects),
        'decision_certified': sum(r['decision_certified'] for r in rows),
        'tasks_complete': len(complete), 'reports_complete': sum(r['report_complete'] for r in rows),
        'unresolved_decisions': sum(not r['decision_certified'] for r in rows),
        'incomplete_reports': sum(not r['report_complete'] for r in rows),
        'execution_failures': sum(r['failure'] is not None for r in rows),
        'unsupported_singletons': sum(r['unsupported_singleton'] for r in rows),
        'objects_with_unsupported_singleton': len({r['obj_id'] for r in rows if r['unsupported_singleton']}),
        'reference_eligible_looks': sum(r['reference_kind'] != 'no_reference' for r in rows),
        'no_reference_looks': sum(r['reference_kind'] == 'no_reference' for r in rows),
        'independently_compared_looks': len(compared), 'independently_compared_objects': len(compared_objects),
        'independent_reference_disagreements': sum(r['reference_agreement'] is False for r in rows),
        'independent_error_objects': len(error_objects),
        'zero_error_object_upper_95_two_sided': 1 - .025 ** (1 / len(compared_objects)) if compared_objects and not error_objects else None,
        'error_bound_scope': 'object any-error event on emitted independently referenced software-policy looks only',
        'mean_imposed_cost_all_requests': float(np.mean(costs)) if known else None,
        'n_known_costs': sum(c is not None for c in costs),
        'completed_task_subset': {'n': len(complete), 'n_objects': len({r['obj_id'] for r in complete}),
            'mean_imposed_cost': float(np.mean([r['imposed_cost_units'] for r in complete])) if complete and all(r['imposed_cost_units'] is not None for r in complete) else None},
        'eligible_intervention_looks': len(eligible), 'eligible_intervention_objects': len({r['obj_id'] for r in eligible})}


def aggregate_grid(rows):
    cells = defaultdict(list)
    for row in rows:
        cells[tuple(row[k] for k in CELL)].append(row)
    return [{**dict(zip(CELL, key)), **aggregate_cell(values)} for key, values in sorted(cells.items())]


def support_contrasts(logical_rows, empirical_rows, *, objects, profiles=('decision_only', 'legacy_report'),
                      replicates=2000, seed=20261102):
    """Explicit empirical-minus-logical EC2 pairing at shared60, all horizons."""
    def eligible(row):
        return row['method'] == 'ec2' and row['cost_scenario'] == 'shared' and row['budget'] == 60 and row['scenario'] == 'clean' and row['missing_semantics'] == 'unknown'
    logical = [r for r in logical_rows if eligible(r)]
    empirical = [r for r in empirical_rows if eligible(r)]
    validate_cartesian(logical, objects=objects, profiles=profiles, budgets=(60,), costs=('shared',), methods=('ec2',), supports=('logical',))
    validate_cartesian(empirical, objects=objects, profiles=profiles, budgets=(60,), costs=('shared',), methods=('ec2',), supports=('empirical',))
    output = []
    for profile in profiles:
        left = {(r['obj_id'], float(r['horizon_days'])): r for r in logical if r['profile'] == profile}
        right = {(r['obj_id'], float(r['horizon_days'])): r for r in empirical if r['profile'] == profile}
        keys = sorted(left)
        pairs = [(right[k], left[k]) for k in keys]
        costs_known = all(r['imposed_cost_units'] is not None and b['imposed_cost_units'] is not None for r, b in pairs)
        data = []
        for obj in sorted(objects):
            block = [(right[k], left[k]) for k in keys if k[0] == obj]
            data.append([float(np.mean([r['imposed_cost_units'] - b['imposed_cost_units'] for r, b in block])) if costs_known else 0,
                         float(np.mean([int(r['task_complete']) - int(b['task_complete']) for r, b in block])),
                         float(np.mean([int(r['decision_certified']) - int(b['decision_certified']) for r, b in block]))])
        values = np.asarray(data)
        rng = np.random.default_rng(seed)
        boot = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
        both = [(r, b) for r, b in pairs if r['task_complete'] and b['task_complete']]
        output.append({'profile': profile, 'budget': 60, 'cost_scenario': 'shared', 'scenario': 'clean',
            'method': 'ec2', 'target_support': 'empirical', 'baseline_support': 'logical',
            'n_paired_requests': len(pairs), 'n_paired_objects': len(objects),
            'target': aggregate_cell(list(right.values())), 'baseline': aggregate_cell(list(left.values())),
            'imposed_cost_difference_all_requests': float(values[:, 0].mean()) if costs_known else None,
            'imposed_cost_difference_all_requests_ci95': np.quantile(boot[:, 0], [.025, .975]).tolist() if costs_known else None,
            'completion_rate_difference': float(values[:, 1].mean()),
            'completion_rate_difference_ci95': np.quantile(boot[:, 1], [.025, .975]).tolist(),
            'decision_coverage_difference': float(values[:, 2].mean()),
            'decision_coverage_difference_ci95': np.quantile(boot[:, 2], [.025, .975]).tolist(),
            'both_complete_subset': {'n': len(both), 'n_objects': len({r['obj_id'] for r, _ in both}),
                'imposed_cost_difference': float(np.mean([r['imposed_cost_units'] - b['imposed_cost_units'] for r, b in both])) if both and all(r['imposed_cost_units'] is not None and b['imposed_cost_units'] is not None for r, b in both) else None},
            'bootstrap_replicates': replicates, 'bootstrap_seed': seed,
            'contrast_scope': 'all requested pairs retained; conditional subset explicitly separate'})
    return output


def reconstruct_ec2(rows, *, snapshots, prior):
    """Reconstruct pre-query posterior regimes; verify recorded choices exactly.

    Only attempted prefixes are exposed to the frozen planner. No future mask,
    source status or scientific outcome enters a reconstructed earlier choice.
    """
    from run_partial_benchmark import GroupedPolicyModel, cost_model, intervention
    from run_partial_benchmark import load_snapshot
    from applecider_agents.flare.partial import map_observations
    models, sessions, output, used_snapshots = {}, {}, [], {}
    for row in rows:
        if row['method'] != 'ec2':
            continue
        identity = {k: row[k] for k in LOOK + CELL}
        if row['failure'] is not None or not isinstance(row.get('acquisition_order'), list):
            output.append({**identity, 'reconstructed': False, 'reconstruction_unavailable': True,
                'reason': 'request failure or recorded query prefix unavailable',
                'attempted_query_count': None, 'zero_posterior_fallback_count': None,
                'zero_edge_step_count': None, 'events': []})
            continue
        key = (row['profile'], row['cost_scenario'], row['support'])
        if key not in models:
            models[key] = GroupedPolicyModel(row['profile'], cost_model(row['cost_scenario']), prior=prior, support=row['support'])
        model = models[key]
        look = (row['obj_id'], float(row['horizon_days']))
        if look not in sessions:
            path = Path(snapshots) / f'{look[0]}_h{look[1]:g}.json'
            try:
                sessions[look] = load_snapshot(path, expected_obj=look[0], expected_horizon=look[1])
                used_snapshots[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                output.append({**identity, 'reconstructed': False, 'reconstruction_unavailable': True,
                    'reason': f'captured snapshot unavailable: {type(exc).__name__}',
                    'attempted_query_count': None, 'zero_posterior_fallback_count': None,
                    'zero_edge_step_count': None, 'events': []})
                continue
        session = sessions[look]
        observations, _ = intervention(session, row['scenario'])
        by_tool = {o.tool: o for o in observations}
        observed, acquired, spent, events = {}, [], 0, []
        for action in row.get('acquisition_order', []):
            indices = model.ec2_controls.compatible(observed)
            mass = model.ec2_controls.mass(indices)
            edge = model.ec2_controls.edge_mass(indices)
            chosen = model.ec2_controls.ec2_next(observed, row['budget'] - spent)
            if chosen != action:
                raise ValueError(f'EC2 reconstruction choice differs at {look}, step{len(acquired)}')
            cut = model.ec2_controls.edge_cut(indices, action)
            reason = 'empty_or_zero_weight_posterior' if mass <= 0 else 'zero_cross_class_edge_mass' if edge == 0 else 'positive_edges_zero_cut_fallback' if cut <= 0 else 'positive_cross_class_edge_cut'
            events.append({'step': len(acquired), 'observed_prefix': dict(observed), 'selected_action': action,
                           'posterior_worlds': len(indices), 'reference_weight_mass': mass,
                           'cross_class_edge_mass': edge, 'selected_edge_cut': cut, 'reason': reason})
            spent += model.controls.costs.increment(action, acquired)
            acquired.append(action)
            mapped = map_observations(session.scope, [by_tool[action]], blind=True) if action in by_tool else None
            observed[action] = model.outcome(action, mapped) if mapped is not None else -1
            if row['missing_semantics'] == 'negative':
                from run_partial_benchmark import decode_outcome
                observed[action] = decode_outcome(observed[action], len(model.groups[action]))[0]
        output.append({**identity, 'reconstructed': True, 'reconstruction_unavailable': False,
            'attempted_query_count': len(events),
            'zero_posterior_fallback_count': sum(e['reason'] == 'empty_or_zero_weight_posterior' for e in events),
            'zero_edge_step_count': sum(e['reason'] == 'zero_cross_class_edge_mass' for e in events),
            'fallback_count': sum(e['reason'] != 'positive_cross_class_edge_cut' for e in events), 'events': events})
    return {'traces': output, 'used_snapshot_sha256': used_snapshots,
        'used_prior_sha256': hashlib.sha256(Path(prior).read_bytes()).hexdigest(),
        'source_sha256': analysis_source_hashes()}


def assess_literal(rows, *, snapshots, objects, scenarios=('clean',), horizons=(5, 10, 20, 100)):
    """Reproduce literal support/obligation checks from calls and captured inputs."""
    from run_partial_benchmark import load_snapshot, intervention, successful_data_reference
    from applecider_agents.flare.partial import map_observations, evaluate_flare
    expected = set(product(objects, map(float, horizons), scenarios))
    seen, sessions, references, groups, used_snapshots = set(), {}, {}, defaultdict(list), {}
    for row in rows:
        key = (row['obj_id'], float(row['horizon_days']), row['scenario'])
        if key not in expected or key in seen:
            raise ValueError('duplicate or unexpected literal requested row')
        seen.add(key)
        look = key[:2]
        try:
            if look not in sessions:
                path = Path(snapshots) / f'{look[0]}_h{look[1]:g}.json'
                sessions[look] = load_snapshot(path, expected_obj=look[0], expected_horizon=look[1])
                used_snapshots[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                original = map_observations(sessions[look].scope, sessions[look].observations, blind=True)
                references[look] = successful_data_reference(original.payloads,
                    provenance={t: r.get('provenance', {}) for t, r in original.observations.items()})
            session = sessions[look]
            observations, eligible = intervention(session, row['scenario'])
            called = {entry['tool'] for entry in row['logical_calls']}
            actual = evaluate_flare(session.scope, [o for o in observations if o.tool in called], profile='legacy_report', blind=True)
            report = row['literal_report']
            decision = (report['verdict'], report['priority']) if report else None
            internally_supported = decision is not None and actual.decision_certified and decision == actual.decision
            reference = references[look]
            mismatch = decision != reference if decision is not None and reference is not None else None
            assessment = {'obj_id': row['obj_id'], 'horizon_days': row['horizon_days'], 'scenario': row['scenario'],
                'literal_report_returned': report is not None, 'literal_failure': row['failure'],
                'unsupported_literal_singleton': decision is not None and (not internally_supported or mismatch is True),
                'evidence_obligations_complete_on_called_sources': actual.report_complete,
                'original_reference_available': reference is not None, 'original_reference_mismatch': mismatch,
                'eligible_intervention_tools': list(eligible), 'assessment_failure': None}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            assessment = {'obj_id': row['obj_id'], 'horizon_days': row['horizon_days'], 'scenario': row['scenario'],
                'literal_report_returned': row.get('literal_report') is not None, 'literal_failure': row['failure'],
                'unsupported_literal_singleton': False, 'evidence_obligations_complete_on_called_sources': False,
                'original_reference_available': False, 'original_reference_mismatch': None,
                'eligible_intervention_tools': [], 'assessment_failure': f'{type(exc).__name__}: {exc}'}
        groups[row['scenario']].append(assessment)
    if seen != expected:
        raise ValueError('incomplete literal requested grid')
    summaries = []
    for scenario, values in sorted(groups.items()):
        compared = [r for r in values if r['original_reference_mismatch'] is not None]
        summaries.append({'scenario': scenario, 'n_requested': len(values), 'n_objects': len(objects),
            'literal_reports_returned': sum(r['literal_report_returned'] for r in values),
            'literal_exceptions': sum(r['literal_failure'] is not None for r in values),
            'assessment_failures': sum(r['assessment_failure'] is not None for r in values),
            'unsupported_literal_singletons': sum(r['unsupported_literal_singleton'] for r in values),
            'objects_with_unsupported_literal_singleton': len({r['obj_id'] for r in values if r['unsupported_literal_singleton']}),
            'evidence_obligations_complete_on_called_sources': sum(r['evidence_obligations_complete_on_called_sources'] for r in values),
            'original_reference_comparisons': len(compared),
            'original_reference_mismatches': sum(r['original_reference_mismatch'] is True for r in values),
            'no_reference_looks': sum(not r['original_reference_available'] for r in values),
            'eligible_intervention_looks': sum(bool(r['eligible_intervention_tools']) for r in values),
            'eligible_intervention_objects': len({r['obj_id'] for r in values if r['eligible_intervention_tools']})})
    return {'grid_validation': {'planned_rows': len(expected), 'completed_rows': len(seen), 'unique_complete_grid': True},
        'cells': summaries, 'assessments': [r for values in groups.values() for r in values],
        'used_snapshot_sha256': used_snapshots, 'source_sha256': analysis_source_hashes(),
        'scope': 'matched captured-prefix literal policy; called evidence obligations, not semantic verification of legacy prose or astronomical truth'}


def analysis_source_hashes():
    repo = Path(__file__).resolve().parents[1]
    files = ('scripts/analyze_partial_benchmark.py', 'scripts/run_partial_benchmark.py',
        'scripts/run_legacy_diagnostic.py', 'applecider_agents/core/acquisition.py',
        'applecider_agents/core/partial.py', 'applecider_agents/core/evidence.py',
        'applecider_agents/core/observations.py', 'applecider_agents/core/membership.py',
        'applecider_agents/flare/partial.py')
    return {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in files}


def analytical_report_controls(*, objects, budgets, costs, horizons=(5, 10, 20, 100)):
    """Static feasibility, explicitly not observed execution or scheduling gains."""
    floors = {'shared': 26, 'independent': 48}
    return [{'profile': 'legacy_report', 'cost_scenario': cost, 'budget': budget,
             'control': 'analytical_zero_query_abstention', 'actual_runtime_measured': False,
             'n_requested': len(objects) * len(horizons), 'n_objects': len(objects),
             'completed_task_cost_floor': floors[cost], 'expected_completed_tasks': 0,
             'imposed_acquisition_cost': 0, 'decision_certificate_issued': False,
             'scope': 'all requests below a proved profile/tool-alphabet cost floor; no empirical accuracy claim'}
            for cost in costs for budget in budgets if budget < floors[cost]]


def paired_method_contrasts(rows):
    contrasts = paired_bootstrap(rows)
    by_key = {(r['obj_id'], float(r['horizon_days']), *(r[k] for k in CELL)): r for r in rows}
    for contrast in contrasts:
        objects, both = set(), []
        for row in rows:
            if row['method'] != contrast['method'] or any(row[k] != contrast[k] for k in CELL if k != 'method'):
                continue
            key = (row['obj_id'], float(row['horizon_days']), *(row[k] if k != 'method' else contrast['baseline'] for k in CELL))
            base = by_key[key]
            if row['task_complete'] and base['task_complete']:
                objects.add(row['obj_id'])
                both.append((row, base))
        contrast['n_both_complete'] = len(both)
        contrast['both_complete_effective_objects'] = len(objects)
        contrast['n_both_complete_known_cost'] = sum(r['imposed_cost_units'] is not None and b['imposed_cost_units'] is not None for r, b in both)
        contrast['cost_difference_resolvable_paired_subset'] = float(np.mean([r['imposed_cost_units'] - b['imposed_cost_units'] for r, b in both])) if both and contrast['n_both_complete_known_cost'] == len(both) else None
    return contrasts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('merge', 'support', 'ec2-trace', 'literal'), required=True)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--empirical-inputs', nargs='+', type=Path)
    parser.add_argument('--membership', type=Path, required=True)
    parser.add_argument('--partition', choices=('design', 'tune', 'confirm'), required=True)
    parser.add_argument('--limit', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profiles', default='decision_only,legacy_report')
    parser.add_argument('--cost-scenarios', default='shared,independent')
    parser.add_argument('--budgets', default='60,5,25,40')
    parser.add_argument('--scenarios', default='clean')
    parser.add_argument('--methods', default='eager,compiled,fail_closed,ec2')
    parser.add_argument('--snapshots', type=Path)
    parser.add_argument('--prior', type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('choose a fresh analysis output path')
    source_start = analysis_source_hashes()
    objects = selected_membership(args.membership, args.partition, args.limit)
    rows = [json.loads(line) for path in args.inputs for line in path.read_text().splitlines()] if args.mode == 'literal' else read_rows(args.inputs)
    profiles = args.profiles.split(',')
    if args.mode == 'literal':
        if args.snapshots is None:
            parser.error('literal assessment requires --snapshots')
        result = assess_literal(rows, snapshots=args.snapshots, objects=objects, scenarios=args.scenarios.split(','))
    elif args.mode == 'merge':
        validation = validate_cartesian(rows, objects=objects, profiles=profiles,
            budgets=list(map(int, args.budgets.split(','))), costs=args.cost_scenarios.split(','), methods=args.methods.split(','), scenarios=args.scenarios.split(','))
        result = {'grid_validation': validation, 'cells': aggregate_grid(rows),
                  'paired_contrasts_vs_compiled': paired_method_contrasts(rows),
                  'analytical_report_controls': analytical_report_controls(objects=objects,
                    budgets=list(map(int, args.budgets.split(','))), costs=args.cost_scenarios.split(','))}
    elif args.mode == 'support':
        if not args.empirical_inputs:
            parser.error('support contrast requires --empirical-inputs')
        result = {'support_contrasts': support_contrasts(rows, read_rows(args.empirical_inputs), objects=objects, profiles=profiles)}
    else:
        if args.snapshots is None or args.prior is None:
            parser.error('EC2 reconstruction requires --snapshots and --prior')
        target = [r for r in rows if r['method'] == 'ec2' and r['cost_scenario'] == 'shared' and r['budget'] == 60]
        if args.empirical_inputs:
            target += [r for r in read_rows(args.empirical_inputs) if r['method'] == 'ec2' and r['cost_scenario'] == 'shared' and r['budget'] == 60]
        supports = sorted({r['support'] for r in target})
        validation = validate_cartesian(target, objects=objects, profiles=profiles, budgets=(60,), costs=('shared',), methods=('ec2',), supports=supports)
        result = {'grid_validation': validation, 'ec2_pre_query_reconstruction': reconstruct_ec2(target, snapshots=args.snapshots, prior=args.prior)}
    paths = args.inputs + (args.empirical_inputs or [])
    result['input_artifacts'] = [{'logical_name': f'input_{i}', 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size} for i, p in enumerate(paths)]
    result['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['source_sha256_start'] = source_start
    result['source_sha256_end'] = analysis_source_hashes()
    result['source_unchanged'] = result['source_sha256_start'] == result['source_sha256_end']
    result['membership_file_sha256'] = hashlib.sha256(args.membership.read_bytes()).hexdigest()
    if args.prior is not None:
        result['prior_file_sha256'] = hashlib.sha256(args.prior.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'mode': args.mode}))


if __name__ == '__main__':
    main()
