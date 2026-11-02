"""Small controlled metric fixtures, not astronomical observations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from test_partial_benchmark import ordinary_session

scripts = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
import analyze_partial_benchmark as analysis
import run_partial_benchmark as benchmark
import run_legacy_diagnostic as literal


def row(obj, horizon, *, support='logical', complete=True, cost=10, failure=None, reference=True):
    return dict(obj_id=obj, horizon_days=horizon, profile='decision_only', budget=60,
        scenario='clean', cost_scenario='shared', support=support, missing_semantics='unknown',
        method='ec2', decision_certified=complete, report_complete=complete, task_complete=complete,
        unsupported_singleton=False, failure=failure, reference_kind='successful_data_policy' if reference else 'no_reference',
        reference_agreement=True if reference and complete else None, imposed_cost_units=cost,
        logical_tool_calls=1, acquisition_order=['get_photometry'])


def grids():
    logical, empirical = [], []
    for obj in ('a', 'b'):
        for h in (5, 10, 20, 100):
            logical.append(row(obj, h))
            empirical.append(row(obj, h, support='empirical', complete=obj == 'a', cost=7,
                failure='timeout' if obj == 'b' and h == 5 else None,
                reference=not (obj == 'b' and h == 5)))
    return logical, empirical


def test_support_pairing_keeps_failure_and_no_reference_denominators():
    logical, empirical = grids()
    result = analysis.support_contrasts(logical, empirical, objects=['a', 'b'], profiles=['decision_only'])[0]
    assert result['n_paired_requests'] == 8
    assert result['n_paired_objects'] == 2
    assert result['target']['execution_failures'] == 1
    assert result['target']['no_reference_looks'] == 1
    assert result['imposed_cost_difference_all_requests'] == -3
    assert result['completion_rate_difference'] == -.5
    assert result['both_complete_subset']['n'] == 4
    assert result['both_complete_subset']['n_objects'] == 1
    assert result['bootstrap_replicates'] == 2000
    assert result['bootstrap_seed'] == 20261102


def test_unknown_cost_suppresses_contrast_without_dropping_pair():
    logical, empirical = grids()
    empirical[0]['imposed_cost_units'] = None
    result = analysis.support_contrasts(logical, empirical, objects=['a', 'b'], profiles=['decision_only'])[0]
    assert result['n_paired_requests'] == 8
    assert result['target']['n_known_costs'] == 7
    assert result['imposed_cost_difference_all_requests'] is None
    assert result['imposed_cost_difference_all_requests_ci95'] is None
    assert result['completion_rate_difference'] == -.5
    assert result['both_complete_subset']['n'] == 4
    assert result['both_complete_subset']['imposed_cost_difference'] is None


@pytest.mark.parametrize('mutation', ['drop', 'duplicate', 'wrong_horizon'])
def test_pair_grid_rejects_incomplete_or_duplicate_requests(mutation):
    logical, empirical = grids()
    if mutation == 'drop': empirical.pop()
    elif mutation == 'duplicate': empirical.append(dict(empirical[0]))
    else: empirical[0]['horizon_days'] = 7
    with pytest.raises(ValueError):
        analysis.support_contrasts(logical, empirical, objects=['a', 'b'], profiles=['decision_only'])


def test_static_profile_floor_controls_are_labeled_analytic():
    controls = analysis.analytical_report_controls(objects=['a', 'b'], budgets=[60, 5, 25, 40], costs=['shared', 'independent'])
    assert len(controls) == 5
    assert not any(c['budget'] == 60 for c in controls)
    assert all(c['n_requested'] == 8 and c['expected_completed_tasks'] == 0 and c['imposed_acquisition_cost'] == 0 for c in controls)
    assert all(not c['actual_runtime_measured'] and not c['decision_certificate_issued'] for c in controls)


def test_failed_rows_are_not_fabricated_zero_query_traces(tmp_path):
    failed = row('missing', 5, failure='snapshot unavailable', reference=False, complete=False, cost=0)
    failed.pop('acquisition_order')
    prior = tmp_path / 'unused.npz'
    prior.write_bytes(b'not loaded because the request has no known execution prefix')
    result = analysis.reconstruct_ec2([failed], snapshots=tmp_path, prior=prior)
    trace = result['traces'][0]
    assert trace['reconstruction_unavailable']
    assert trace['attempted_query_count'] is None
    assert trace['zero_posterior_fallback_count'] is None
    assert result['used_snapshot_sha256'] == {}


def test_zero_posterior_trace_is_reconstructed_from_attempted_prefix(ordinary_session, tmp_path):
    session = ordinary_session
    snapshot = tmp_path / 'fixture_h20.json'
    session.save(snapshot)
    policy = benchmark.GroupedPolicyModel('decision_only', benchmark.cost_model('shared')).policy
    weights = np.zeros(len(policy.states)); weights[0] = 1
    prior = tmp_path / 'controlled_prior.npz'
    np.savez(prior, partition=np.array('tune'), predicates=np.array(policy.predicates), weights=weights)
    model = benchmark.GroupedPolicyModel('decision_only', benchmark.cost_model('shared'), prior=prior, support='empirical')
    result = benchmark.run_case(session, model, method='ec2', budget=60)
    result.update(cost_scenario='shared', support='empirical')
    reconstructed = analysis.reconstruct_ec2([result], snapshots=tmp_path, prior=prior)
    trace = reconstructed['traces'][0]
    assert trace['zero_posterior_fallback_count'] > 0
    assert trace['attempted_query_count'] == result['logical_tool_calls']
    assert [e['selected_action'] for e in trace['events']] == result['acquisition_order']
    assert trace['events'][0]['observed_prefix'] == {}
    assert reconstructed['used_snapshot_sha256']['fixture_h20.json'] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert reconstructed['used_prior_sha256'] == hashlib.sha256(prior.read_bytes()).hexdigest()


def test_literal_assessment_replays_only_called_sources(ordinary_session, tmp_path):
    ordinary_session.save(tmp_path / 'fixture_h20.json')
    code, _ = literal.pinned_function(Path(__file__).parents[1])
    rows = []
    for scenario in ('clean', 'missing_photometry'):
        observations, _ = benchmark.intervention(ordinary_session, scenario)
        result = literal.replay(ordinary_session.model_copy(update={'observations': observations}), code)
        result['scenario'] = scenario; rows.append(result)
    result = analysis.assess_literal(rows, snapshots=tmp_path, objects=['fixture'], scenarios=['clean', 'missing_photometry'], horizons=[20])
    cells = {r['scenario']: r for r in result['cells']}
    assert result['grid_validation']['completed_rows'] == 2
    assert cells['clean']['unsupported_literal_singletons'] == 0
    assert cells['clean']['evidence_obligations_complete_on_called_sources'] == 1
    assert cells['missing_photometry']['unsupported_literal_singletons'] == 1
    assert cells['missing_photometry']['evidence_obligations_complete_on_called_sources'] == 0
    assert cells['missing_photometry']['original_reference_mismatches'] == 1


def test_confirmation_public_runner_requires_configuration_only(tmp_path, monkeypatch):
    config = tmp_path / 'config.json'; config.write_text('{}')
    seen = []
    monkeypatch.setattr(benchmark, 'selected_membership', lambda p, partition, limit: seen.append((partition, limit)) or [])
    monkeypatch.setattr(benchmark, 'load_frozen_membership', lambda p: {'raw_sha256': 'fixture', 'semantic_row_sha256': 'fixture'})
    benchmark.main(['--snapshots', str(tmp_path), '--membership', str(tmp_path / 'fixture.json'),
        '--output', str(tmp_path / 'result'), '--partition', 'confirm', '--limit', '1',
        '--frozen-config', str(config), '--profiles', 'decision_only', '--methods', 'compiled',
        '--cost-scenarios', 'shared', '--budgets', '60'])
    assert seen == [('confirm', 1)]


def test_literal_partition_defaults_and_mask_grid(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(literal, 'selected_membership', lambda p, partition, limit: seen.append((partition, limit)) or [])
    membership = tmp_path / 'fixture.json'; membership.write_text('{}')
    for partition, default_limit in [('design', 96), ('confirm', 1024)]:
        output = tmp_path / partition
        literal.main(['--snapshots', str(tmp_path), '--membership', str(membership), '--output', str(output),
            '--partition', partition, '--scenarios', 'clean,missing_photometry'])
        manifest = json.loads((output / 'manifest.json').read_text())
        assert manifest['partition'] == partition
        assert manifest['scenarios'] == ['clean', 'missing_photometry']
        assert seen[-1] == (partition, default_limit)


def test_public_literal_cli_rows_feed_public_assessment(ordinary_session, tmp_path, monkeypatch):
    from applecider_agents.core.evidence import EvidenceScope
    from applecider_agents.core.observations import RecordedSession
    snapshots = tmp_path / 'snapshots'; snapshots.mkdir()
    for horizon in (5, 10, 20, 100):
        scope = EvidenceScope(obj_id='fixture', horizon_days=horizon, session_id=f'fixture{horizon}')
        observations = []
        for observation in ordinary_session.observations:
            payload = dict(observation.payload) if observation.payload is not None else None
            if payload is not None and 'horizon_days' in payload:
                payload['horizon_days'] = horizon
            observations.append(observation.model_copy(update={'scope': scope,
                'inputs': {'obj_id': 'fixture', 'horizon_days': horizon}, 'payload': payload}))
        RecordedSession(scope=scope, observations=tuple(observations)).save(snapshots / f'fixture_h{horizon}.json')
    monkeypatch.setattr(literal, 'selected_membership', lambda *args: ['fixture'])
    membership = tmp_path / 'membership.json'; membership.write_text('{}')
    output = tmp_path / 'literal'
    literal.main(['--snapshots', str(snapshots), '--membership', str(membership), '--output', str(output),
                  '--scenarios', 'clean,missing_photometry'])
    rows = [json.loads(line) for line in (output / 'rows.jsonl').read_text().splitlines()]
    assert len(rows) == 8
    assert {r['scenario'] for r in rows} == {'clean', 'missing_photometry'}
    result = analysis.assess_literal(rows, snapshots=snapshots, objects=['fixture'], scenarios=['clean', 'missing_photometry'])
    assert result['grid_validation']['unique_complete_grid']
    assert result['grid_validation']['completed_rows'] == 8
    assert not any(c['assessment_failures'] for c in result['cells'])


def test_method_subset_denominators_include_unknown_cost_pairs():
    rows = []
    for obj in ('a', 'b'):
        for horizon in (5, 10, 20, 100):
            baseline = row(obj, horizon, cost=None if obj == 'a' and horizon == 5 else 10)
            baseline['method'] = 'compiled'
            rows.extend([baseline, row(obj, horizon, cost=7)])
    result = analysis.paired_method_contrasts(rows)[0]
    assert result['n_pairs'] == 8
    assert result['n_both_complete'] == 8
    assert result['both_complete_effective_objects'] == 2
    assert result['n_both_complete_known_cost'] == 7
    assert result['cost_difference_resolvable_paired_subset'] is None
