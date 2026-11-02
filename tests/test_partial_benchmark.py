"""Controlled software checks; fixtures are not real astronomical observations."""
import importlib.util
from pathlib import Path
import sys

import pytest
from applecider_agents.core.evidence import EvidenceScope, ToolObservation
from applecider_agents.core.observations import RecordedSession

spec = importlib.util.spec_from_file_location('partial_benchmark', Path(__file__).parents[1] / 'scripts' / 'run_partial_benchmark.py')
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


@pytest.fixture
def ordinary_session():
    scope = EvidenceScope(obj_id='fixture', horizon_days=20, session_id='fixture')
    payloads = {
        'flare_classify': dict(obj_id='fixture', predicted='SN_Ia', prediction_set=['SN_Ia'],
            credibility=.5, argmax_excluded_from_set=False, model='fixture', probabilities={'SN_Ia': .8}),
        'flare_anomaly': dict(obj_id='fixture', calibrated=True, alarms={'1%': False, '5%': False, '10%': False},
            alarm_far=None, alarm_score='fixture', alarm_p=.5, calibration='fixture', energy_percentile=70),
        'get_context': dict(obj_id='fixture', host=None, redshift={'z_phot': None},
            flags=dict(stellar_astrometry=False, nuclear=False, agn_mid_ir_colour=False,
                       superluminous_if_host_is_right=False, offset_from_host_gt_10arcsec=False)),
        'get_photometry': dict(obj_id='fixture', n_detections=8, horizon_days=20,
            span_days=20, bands={'g': {'peak_day': 10}}, peak_mag_observed=19),
    }
    rows = [ToolObservation(tool=tool, scope=scope, inputs={'obj_id': 'fixture', 'horizon_days': 20},
            status='success', payload=payload) for tool, payload in payloads.items()]
    rows.append(ToolObservation(tool='light_curve_history', scope=scope,
                inputs={'obj_id': 'fixture'}, status='unavailable', availability={'no_search_performed': True}))
    return RecordedSession(scope=scope, observations=tuple(rows))


def test_profile_fairness_and_literal_reference(ordinary_session):
    model = benchmark.GroupedPolicyModel('legacy_report', benchmark.cost_model('shared'))
    for method in ('eager', 'compiled', 'fail_closed', 'ec2'):
        row = benchmark.run_case(ordinary_session, model, method=method, budget=40)
        assert row['reference'] == ('ordinary', 0)
        assert row['task_complete'], (method, row)
        assert not row['unsupported_singleton']
        assert {'flare_classify', 'flare_anomaly', 'get_context'} <= set(row['acquisition_order'])
    decision = benchmark.GroupedPolicyModel('decision_only', benchmark.cost_model('shared'))
    report = benchmark.run_case(ordinary_session, model, method='compiled', budget=3)
    cheap = benchmark.run_case(ordinary_session, decision, method='compiled', budget=3)
    assert not report['task_complete']
    assert not cheap['task_complete']


def test_missing_history_is_not_mask_eligible(ordinary_session):
    rows, eligible = benchmark.intervention(ordinary_session, 'missing_history')
    assert not eligible
    assert next(o for o in rows if o.tool == 'light_curve_history').payload is None
    rows, eligible = benchmark.intervention(ordinary_session, 'missing_context_history')
    assert eligible == ('get_context',)
    assert next(o for o in rows if o.tool == 'get_context').status == 'unavailable'
    assert next(o for o in ordinary_session.observations if o.tool == 'get_context').status == 'success'


def test_report_missing_classification_stays_in_denominator(ordinary_session):
    model = benchmark.GroupedPolicyModel('legacy_report', benchmark.cost_model('shared'))
    row = benchmark.run_case(ordinary_session, model, method='compiled', budget=40, scenario='missing_classify')
    assert not row['task_complete']
    assert row['eligible_mask'] == ['flare_classify']
    row.update(cost_scenario='shared', support='logical')
    summary = benchmark.summarize([row])[0]
    assert summary['n_requested'] == 1
    assert summary['tasks_complete'] == 0
    assert summary['n_objects'] == 1


def test_prior_does_not_control_logical_admissibility():
    model = benchmark.GroupedPolicyModel('decision_only', benchmark.cost_model('shared'))
    # Zero empirical mass on every rare state cannot certify an empty record.
    model.controls.weights[:] = 0
    model.controls.weights[0] = 1
    assert model._decision(()) is None
    assert model.controls.ec2_next({}, 40) is not None


def test_paired_bootstrap_clusters_by_object_and_retains_failures():
    rows = []
    for obj in ('a', 'b'):
        for horizon in (5, 10, 20, 100):
            for method, cost, complete in (('compiled', 10, True), ('ec2', 7, obj == 'a')):
                rows.append(dict(obj_id=obj, horizon_days=horizon, method=method, imposed_cost_units=cost,
                    task_complete=complete, profile='legacy_report', budget=40, scenario='clean',
                    cost_scenario='shared', support='logical', missing_semantics='unknown'))
    result = benchmark.paired_bootstrap(rows)[0]
    assert result['n_pairs'] == 8
    assert result['n_objects'] == 2
    assert result['cost_difference_all_requests'] == -3
    assert result['completion_rate_difference'] == -.5
    assert result['n_both_complete'] == 4
    assert result['replicates'] == 2000


def test_independent_legacy_fixture_at_boundaries(ordinary_session):
    payloads = {o.tool: o.payload for o in ordinary_session.observations if o.payload is not None}
    assert benchmark.successful_data_reference(payloads) == ('ordinary', 0)
    payloads['get_photometry']['n_detections'] = 7
    assert benchmark.successful_data_reference(payloads) == ('insufficient_data', 0)


def test_missing_report_field_cannot_be_planner_terminal(ordinary_session):
    from applecider_agents.flare.partial import map_observations, evaluate_flare
    rows = []
    for observation in ordinary_session.observations:
        if observation.tool == 'flare_anomaly':
            payload = dict(observation.payload)
            del payload['energy_percentile']
            observation = observation.model_copy(update={'payload': payload})
        rows.append(observation)
    mapped = map_observations(ordinary_session.scope, rows, blind=True)
    model = benchmark.GroupedPolicyModel('legacy_report', benchmark.cost_model('shared'))
    observed = {a: model.outcome(a, mapped) for a in benchmark.ACTIONS}
    assert not model.certificate(observed)
    assert not evaluate_flare(ordinary_session.scope, rows, profile='legacy_report', blind=True).task_complete
    assert 'flare_anomaly' in model.required(observed)


def test_raw_missing_detection_count_is_no_reference(ordinary_session):
    payloads = {o.tool: dict(o.payload) for o in ordinary_session.observations if o.payload is not None}
    del payloads['get_photometry']['n_detections']
    assert benchmark.successful_data_reference(payloads) is None


def test_public_typed_snapshot_is_accepted(ordinary_session, tmp_path):
    path = tmp_path / 'session.json'
    ordinary_session.save(path)
    loaded = benchmark.load_snapshot(path)
    assert loaded.scope == ordinary_session.scope
    assert loaded.observations == ordinary_session.observations


def test_zero_error_bound_uses_objects_not_horizon_rows():
    rows = []
    for h in (5, 10, 20, 100):
        row = benchmark.failed_request_row('one-object', h, 'legacy_report', 'compiled', 40, 'clean', 'unknown', 'fixture')
        row.update(cost_scenario='shared', support='logical')
        rows.append(row)
    result = benchmark.summarize(rows)[0]
    assert result['snapshot_error_count_descriptive'] == '0/4'
    assert result['n_objects'] == 1
    assert result['zero_error_object_two_sided_95_upper'] == pytest.approx(.975)


@pytest.mark.parametrize('credibility', [None, 'invalid', float('nan')])
def test_unknown_classification_is_no_reference(ordinary_session, credibility):
    payloads = {o.tool: dict(o.payload) for o in ordinary_session.observations if o.payload is not None}
    payloads['flare_classify']['credibility'] = credibility
    assert benchmark.successful_data_reference(payloads) is None


def test_invalid_peak_day_is_no_reference(ordinary_session):
    payloads = {o.tool: dict(o.payload) for o in ordinary_session.observations if o.payload is not None}
    payloads['get_photometry']['bands'] = {'g': {'peak_day': 'invalid'}}
    assert benchmark.successful_data_reference(payloads) is None


def test_relevant_history_requires_measured_or_completed_source(ordinary_session):
    payloads = {o.tool: dict(o.payload) for o in ordinary_session.observations if o.payload is not None}
    payloads['flare_classify']['prediction_set'] = ['SN_Ia', 'AGN']
    payloads['get_context']['flags'] = dict(payloads['get_context']['flags'], nuclear=True)
    payloads['light_curve_history'] = {'found': True, 'n_detections': 20}
    assert benchmark.successful_data_reference(payloads) is None
    payloads['light_curve_history'] = {'found': False}
    assert benchmark.successful_data_reference(payloads) is None
    assert benchmark.successful_data_reference(payloads,
        provenance={'light_curve_history': {'completed_search': True}}) == ('ordinary', 0)


def test_primary_error_gate_sees_independent_reference_mismatch(ordinary_session, monkeypatch):
    import applecider_agents.flare.partial as partial
    original = partial.evaluate_flare
    from applecider_agents.core.partial import Decision
    def corrupted(*args, **kwargs):
        result = original(*args, **kwargs)
        return result.model_copy(update={'status': 'decided', 'decision_certified': True,
            'possible_decisions': (Decision('anomaly_review', 1),)})
    monkeypatch.setattr(partial, 'evaluate_flare', corrupted)
    model = benchmark.GroupedPolicyModel('decision_only', benchmark.cost_model('shared'))
    row = benchmark.run_case(ordinary_session, model, method='eager', budget=40)
    assert row['engine_certificate_valid']
    assert row['independent_reference_mismatch']
    assert row['unsupported_singleton']
    assert not row['decision_certified']
    assert not row['task_complete']


def test_snapshot_header_cannot_replace_frozen_request(ordinary_session, tmp_path):
    path = tmp_path / 'wrong_header.json'
    ordinary_session.save(path)
    with pytest.raises(ValueError, match='frozen requested identity'):
        benchmark.load_snapshot(path, expected_obj='different', expected_horizon=20)
    with pytest.raises(ValueError, match='frozen requested horizon'):
        benchmark.load_snapshot(path, expected_obj='fixture', expected_horizon=5)


def test_ec2_failed_evidence_marginalizes_science_but_remains_attempted():
    import numpy as np
    model = benchmark.GroupedPolicyModel('legacy_report', benchmark.cost_model('shared'))
    with_failure = {'flare_classify': -1, 'get_photometry': 1}
    posterior = model.ec2_controls.compatible(with_failure)
    expected = model.ec2_controls.compatible({'get_photometry': 1})
    assert len(posterior) > 0
    assert np.array_equal(posterior, expected)
    assert model.ec2_controls.ec2_next(with_failure, 40) != 'flare_classify'
    assert not model.certificate(with_failure)


def test_ec2_report_invalid_keeps_valid_predicate_bits():
    import numpy as np
    model = benchmark.GroupedPolicyModel('legacy_report', benchmark.cost_model('shared'))
    valid = {'flare_anomaly': 1}
    invalid_report = {'flare_anomaly': 1 | (1 << len(model.groups['flare_anomaly']))}
    assert np.array_equal(model.ec2_controls.compatible(valid), model.ec2_controls.compatible(invalid_report))
    assert 'flare_anomaly' in model.required(invalid_report)
    # Generic finite joint outcome semantics are deliberately unchanged.
    assert len(model.controls.compatible(invalid_report)) == 0


@pytest.mark.parametrize('scenario', ['wrong_object', 'wrong_horizon', 'stale_session'])
def test_scope_faults_retain_original_payload_and_reject_binding(ordinary_session, scenario):
    from applecider_agents.flare.partial import map_observations
    original = next(o for o in ordinary_session.observations if o.tool == 'flare_classify')
    transformed, eligible = benchmark.intervention(ordinary_session, scenario)
    assert eligible == ('flare_classify',)
    changed = next(o for o in transformed if o.tool == 'flare_classify')
    assert changed.payload == original.payload
    assert changed.status == 'success'
    assert changed.scope != ordinary_session.scope
    assert original.scope == ordinary_session.scope
    mapped = map_observations(ordinary_session.scope, transformed, blind=True)
    assert 'flare_classify' not in mapped.usable_tools
    assert not any(p in mapped.constraints for p in benchmark.GroupedPolicyModel(
        'decision_only', benchmark.cost_model('shared')).groups['flare_classify'])


def test_native_typed_collector_cost_receipts_are_preserved(ordinary_session):
    observations = list(ordinary_session.observations)
    observations[0] = observations[0].model_copy(update={'provenance': {
        'source': 'actual_local_upstream_tool', 'wall_seconds': .75,
        'backend_work_delta': {'feature_calls': 1, 'inference_calls': 1}}})
    session = ordinary_session.model_copy(update={'observations': tuple(observations), 'costs': {
        'kind': 'actual_local_acquisition', 'wall_seconds': 1.25, 'logical_calls': 4,
        'feature_calls': 1, 'inference_calls': 1,
        'backend_records': [{'backend': 'classify', 'phase_classifier_cache': 'warm', 'wall_seconds': .5}]}})
    receipt = benchmark.recorded_work(session)
    assert receipt['source_acquisition_wall_seconds'] == 1.25
    assert receipt['source_observation_wall_seconds'] == .75
    assert receipt['source_backend_work_delta'] == {'feature_calls': 1, 'inference_calls': 1}
    assert receipt['source_backend_records'] == session.costs['backend_records']
    assert receipt['source_session_costs'] == session.costs


def test_missing_cost_receipt_is_unknown_not_fabricated_zero(ordinary_session):
    receipt = benchmark.recorded_work(ordinary_session)
    assert receipt['source_acquisition_wall_seconds'] is None
    assert receipt['source_observation_wall_seconds'] is None
    assert receipt['source_backend_work_delta'] is None
    assert receipt['source_backend_records'] is None


def partial_photometry_session(ordinary_session):
    observations = []
    for observation in ordinary_session.observations:
        if observation.tool == 'get_photometry':
            observation = observation.model_copy(update={'payload': dict(observation.payload,
                n_detections=0, bands={}, peak_mag_observed=None)})
        observations.append(observation)
    return ordinary_session.model_copy(update={'observations': tuple(observations)})


def test_ec2_conditions_on_each_known_partial_group_bit(ordinary_session):
    import numpy as np
    from applecider_agents.flare.partial import map_observations
    session = partial_photometry_session(ordinary_session)
    mapped = map_observations(session.scope, session.observations, blind=True)
    model = benchmark.GroupedPolicyModel('decision_only', benchmark.cost_model('shared'))
    outcome = model.outcome('get_photometry', mapped)
    assert isinstance(outcome, tuple)
    assert model.constraints({'get_photometry': outcome}) == {'enough_detections': False, 'young': False}
    posterior = model.ec2_controls.compatible({'get_photometry': outcome})
    assert len(posterior) > 0
    assert np.all(model.policy.states[posterior, model.columns['enough_detections']] == False)
    assert np.all(model.policy.states[posterior, model.columns['young']] == False)
    assert len(np.unique(model.policy.states[posterior, model.columns['bright']])) == 2
    assert model.certificate({'get_photometry': outcome})
    assert len(model.controls.compatible({'get_photometry': outcome})) == 0


def test_tune_order_scores_actual_partial_payload_completion(ordinary_session, tmp_path):
    session = partial_photometry_session(ordinary_session)
    output = tmp_path / 'fixture_tune_config.json'
    config = benchmark.fit_tune_configuration([session], output,
        budgets=[3], cost_scenarios=['shared'], profiles=['decision_only'])
    score = config['tune_order_scores']['decision_only|shared|3']
    assert score['n_complete'] == 1
    assert score['total_spent'] == 3
    model = benchmark.GroupedPolicyModel('decision_only', benchmark.cost_model('shared'))
    row = benchmark.run_case(session, model, method='compiled', budget=3,
        order=config['compiled_orders']['decision_only|shared|3'])
    assert row['task_complete']
    assert row['decision'] == ('insufficient_data', 0)
    assert row['imposed_cost_units'] == score['total_spent']


@pytest.mark.parametrize('profile', ['decision_only', 'legacy_report'])
def test_unavailable_history_preserves_clean_finite_outcome(ordinary_session, profile):
    from applecider_agents.flare.partial import map_observations
    unavailable = next(o for o in ordinary_session.observations if o.tool == 'light_curve_history')
    mapping = map_observations(ordinary_session.scope, [unavailable], blind=True)
    model = benchmark.GroupedPolicyModel(profile, benchmark.cost_model('shared'))
    assert model.outcome('light_curve_history', mapping) == -1
    assert len(model.controls.compatible({'light_curve_history': -1})) == len(model.controls.compatible({}))


def test_portable_frozen_cohort_has_same_selected_objects():
    import json
    from applecider_agents.core.membership import load_frozen_membership
    repo = Path(__file__).parents[1]
    portable = repo / 'benchmarks' / 'evidence_policy_membership.json'
    membership = load_frozen_membership(portable)
    expected = [r['obj_id'] for r in sorted(
        (r for r in membership['rows'] if r['partition'] == 'design'), key=lambda r: r['order_key'])[:96]]
    assert benchmark.selected_membership(portable, 'design', 96) == expected
    assert len(membership['rows']) == 8922
