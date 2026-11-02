"""Offline observation ownership and prefix regressions; no live services or labels."""
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from applecider_agents.core import data as DATA, object_tools as O
from applecider_agents.core.observations import load_raw_snapshot, RecordedSession
from applecider_agents.flare import tools as T
from applecider_agents.flare.observations import collect_flare


def events(peak=2.0, tail=4.0):
    array = np.zeros((10, 15))
    array[:, 0] = np.arange(10)
    array[:, 2] = np.arange(10) % 2
    array[:, 3] = peak
    array[-1, 3] = tail
    array[:, 4] = .05
    array[:, 14] = 12345  # Test-only truth sentinel, never a scientific feature.
    return array


@pytest.fixture
def fake_science(monkeypatch):
    def frame(ev, ctx, z):
        return pd.DataFrame({'marker': [float(ev[:, 3].max())]}), 0, ctx
    def classify(X, base_rate, horizon=None):
        return dict(model='FIXTURE', model_horizon_days=horizon, predicted='SN_Ia',
            probabilities={'SN_Ia': 1.0}, probability_intervals={}, prediction_set=['SN_Ia'],
            alpha=.1, p_values={'SN_Ia': .9}, credibility=.9, confidence=.9,
            argmax_excluded_from_set=False, coverage_of_predicted=.9, coverage_of_predicted_err=.01,
            anomaly={'energy_percentile': float(X.marker.iloc[0]), 'alarm_far': None,
                     'alarms': {'1%': False}, 'alarm_p': .9, 'calibrated': True, 'alarm_score': 'FIXTURE'})
    monkeypatch.setattr(T, 'feature_frame', frame)
    monkeypatch.setattr(T, 'classify', classify)
    monkeypatch.setattr(DATA, 'history', lambda obj: pytest.fail('offline collector called history'))


def test_replay_captures_status_scope_and_no_live_cost(tmp_path):
    path = tmp_path / 'snapshot.json'
    raw = {'obj_id': 'FIXTURE', 'horizon_days': 5, 'wall_seconds': 10,
           'scientific_input': {'event_columns': ['dt'], 'events': [[0]]}, 'observations': [
        {'tool': 'flare_classify', 'input': {'obj_id': 'FIXTURE', 'horizon_days': 5},
         'status': 'success', 'result': {'obj_id': 'FIXTURE', 'credibility': .7}},
        {'tool': 'get_context_prefix', 'input': {'obj_id': 'FIXTURE'}, 'status': 'success',
         'effective_photometric_horizon_days': 5, 'result': {'flags': {}}},
        {'tool': 'get_context_legacy100', 'input': {'obj_id': 'FIXTURE'}, 'status': 'success',
         'effective_photometric_horizon_days': 100, 'result': {'flags': {}}},
        {'tool': 'light_curve_history', 'input': {'obj_id': 'FIXTURE'}, 'status': 'unavailable', 'result': None}]}
    path.write_text(json.dumps(raw))
    session = load_raw_snapshot(path)
    assert len(session.observations) == 3
    assert session.costs['kind'] == 'recorded_acquisition_not_live_replay'
    assert session.observations[-1].payload is None
    legacy = load_raw_snapshot(path, context_variant='legacy100')
    assert next(o for o in legacy.observations if o.tool == 'get_context').scope.horizon_days == 100
    restored = RecordedSession.model_validate_json(session.model_dump_json())
    assert restored == session


def test_seed_replacement_and_input_mutation_cannot_reuse_science(fake_science):
    original = events(2, 2)
    DATA.seed('FIXTURE', original, {'host_sep': 1})
    first = T._flare('FIXTURE', 5)['anomaly']['energy_percentile']
    original[:, 3] = 9  # seed owns a copy.
    assert T._flare('FIXTURE', 5)['anomaly']['energy_percentile'] == first == 2
    DATA.seed('FIXTURE', events(3, 3), {'host_sep': 2})
    assert T._flare('FIXTURE', 5)['anomaly']['energy_percentile'] == 3
    DATA.unseed('FIXTURE')


def test_optional_context_horizon_preserves_legacy_default(monkeypatch):
    calls = []
    def prepared(obj, horizon):
        calls.append(horizon)
        return None, 19, {'_source': 'provided'}
    monkeypatch.setattr(O, '_events_and_context', prepared)
    O.get_context.call({'obj_id': 'FIXTURE'})
    O.get_context.call({'obj_id': 'FIXTURE', 'horizon_days': 5})
    assert calls == [100, 5]


def test_collector_future_tail_context_and_shared_backend(fake_science):
    first = collect_flare('FIXTURE', 5, events_array=events(tail=4), context={'z_phot': .1, 'host_sep': 1})
    second = collect_flare('FIXTURE', 5, events_array=events(tail=9), context={'z_phot': .1, 'host_sep': 1})
    assert first.scope.session_id != second.scope.session_id
    for session in (first, second):
        assert session.costs['inference_calls'] == session.costs['feature_calls'] == 1
        assert session.costs['logical_calls'] == 4
        assert session.observations[-1].status == 'unavailable'
        assert 'label' not in session.scientific_input['event_columns']
        assert all(len(row) == 14 for row in session.scientific_input['events'])
        anomaly = next(o for o in session.observations if o.tool == 'flare_anomaly')
        assert anomaly.provenance['backend_work_delta']['inference_calls'] == 0
    context1 = next(o for o in first.observations if o.tool == 'get_context').payload
    context2 = next(o for o in second.observations if o.tool == 'get_context').payload
    assert context1 == context2
    assert 'FIXTURE' not in DATA._SEEDED


def test_failed_inputs_remain_errors_not_zero_science():
    session = collect_flare('FIXTURE', 5, events_array=np.zeros((2, 3)), context={})
    assert [o.status for o in session.observations] == ['error'] * 4 + ['unavailable']
    assert all(o.payload is None for o in session.observations)
    assert session.costs['inference_calls'] == 0


def test_parallel_same_object_sessions_are_isolated(fake_science):
    def run(value):
        session = collect_flare('FIXTURE', 5, events_array=events(value, value), context={'host_sep': value})
        return next(o for o in session.observations if o.tool == 'flare_anomaly').payload['energy_percentile']
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(run, [2, 3, 4, 5])) == [2, 3, 4, 5]


def test_legacy_reader_and_two_new_sessions_cannot_mix_data(fake_science, monkeypatch):
    import threading
    barrier = threading.Barrier(3)
    def frame(ev, ctx, z):
        expected = float(ev[:, 3].max())
        barrier.wait(timeout=5)
        # Read again after all3 legacy/new calls are simultaneously in flight.
        reread = DATA.events('FIXTURE', 5)
        assert float(reread[:, 3].max()) == expected
        assert DATA.context('FIXTURE')['host_sep'] == ctx['host_sep'] == expected
        return pd.DataFrame({'marker': [expected]}), 0, ctx
    monkeypatch.setattr(T, 'feature_frame', frame)
    DATA.seed('FIXTURE', events(7, 7), {'host_sep': 7})
    def new(value):
        result = collect_flare('FIXTURE', 5, events_array=events(value, value), context={'host_sep': value})
        assert result.costs['feature_calls'] == result.costs['inference_calls'] == 1
        return next(o for o in result.observations if o.tool == 'flare_anomaly').payload['energy_percentile']
    with ThreadPoolExecutor(max_workers=3) as pool:
        old = pool.submit(lambda: T._flare('FIXTURE', 5)['anomaly']['energy_percentile'])
        a = pool.submit(new, 2)
        b = pool.submit(new, 3)
        assert [old.result(), a.result(), b.result()] == [7, 2, 3]
    assert DATA.context('FIXTURE')['host_sep'] == 7
    DATA.unseed('FIXTURE')


def test_invalid_future_science_cannot_break_valid_prefix(fake_science):
    clean = events()
    corrupted = clean.copy()
    corrupted[-1, 2:14] = np.nan
    a = collect_flare('FIXTURE', 5, events_array=clean, context={'host_sep': 1})
    b = collect_flare('FIXTURE', 5, events_array=corrupted, context={'host_sep': 1})
    assert [o.status for o in a.observations] == [o.status for o in b.observations]
    assert [o.payload for o in a.observations] == [o.payload for o in b.observations]


def test_local_tables_without_matching_science_are_unavailable(fake_science, tmp_path):
    pd.DataFrame({'obj_id': ['OTHER'], 'host_sep': [1.0]}).to_csv(tmp_path / 'hosts.csv', index=False)
    session = collect_flare('FIXTURE', 5, events_array=events(), data_dir=tmp_path)
    context = next(o for o in session.observations if o.tool == 'get_context')
    assert context.status == 'unavailable' and context.payload is None
    assert session.metadata['context_files'] == ['hosts.csv']
    assert session.metadata['context_matched_files'] == []
    pd.DataFrame({'obj_id': ['FIXTURE'], 'host_sep': [np.nan]}).to_csv(tmp_path / 'hosts.csv', index=False)
    session = collect_flare('FIXTURE', 5, events_array=events(), data_dir=tmp_path)
    context = next(o for o in session.observations if o.tool == 'get_context')
    assert context.status == 'unavailable' and context.payload is None
    assert session.metadata['context_matched_files'] == ['hosts.csv']


def test_opt_in_cli_is_offline_partial_and_never_publishes(monkeypatch, capsys, tmp_path):
    from applecider_agents.cli import main
    from applecider_agents.core import skyportal_client
    monkeypatch.setattr(skyportal_client, 'publish', lambda *a, **kw: pytest.fail('partial CLI published'))
    monkeypatch.setattr(DATA.C, 'EVENT_DIRS', [tmp_path])
    main(['partial', 'MISSING_LOCAL_FIXTURE', '--horizon', '5'])
    output = json.loads(capsys.readouterr().out)
    assert output['result']['status'] == 'unresolved'
    assert not output['result']['report_complete']
    assert output['costs']['inference_calls'] == 0
    assert output['metadata']['history'] == 'unavailable_offline'


def test_mcp_opt_in_tool_is_registered_and_remains_nonposting(monkeypatch, tmp_path):
    import asyncio
    from applecider_agents.core.mcp import build
    from applecider_agents.core import skyportal_client
    monkeypatch.setattr(skyportal_client, 'publish', lambda *a, **kw: pytest.fail('partial MCP published'))
    monkeypatch.setattr(DATA.C, 'EVENT_DIRS', [tmp_path])
    server = build(['flare'])
    assert 'flare_partial_triage' in {t.name for t in asyncio.run(server.list_tools())}
    result = asyncio.run(server.call_tool('flare_partial_triage', {'obj_id': 'MISSING_LOCAL_FIXTURE', 'horizon_days': 5}))
    payload = getattr(result, 'structured_content', None) or json.loads(result.content[0].text)
    payload = payload.get('result', payload)
    # MCP may wrap the callable's entire return dictionary in an outer result.
    decision = payload.get('result', payload)
    assert decision['status'] == 'unresolved'
    assert decision['report_complete'] is False


def test_portable_prepare_retains_every_failed_request(tmp_path):
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_partial_benchmark.py'
    spec = importlib.util.spec_from_file_location('prepare_partial_benchmark_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    membership = tmp_path / 'membership.json'
    membership.write_text(json.dumps({'rows': [{'obj_id': 'MISSING_FIXTURE', 'partition': 'design', 'order_key': 'a', 'truth': 'DO_NOT_EXPORT'}]}))
    out = tmp_path / 'out'
    exit_code = module.main(['--membership', str(membership), '--n', '1', '--horizons', '5', '10',
                             '--events-dir', str(tmp_path), '--data-dir', str(tmp_path), '--out', str(out)])
    assert exit_code == 1
    receipt = json.loads((out / 'receipt.json').read_text())
    assert receipt['planned_sessions'] == receipt['terminal_records'] == 2
    assert receipt['status_counts'] == {'captured_with_error': 2}
    assert len((out / 'denominator.jsonl').read_text().splitlines()) == 4
    assert 'DO_NOT_EXPORT' not in (out / 'sessions.jsonl').read_text()


def test_replay_source_failure_exports_unknown_session_not_only_ledger(tmp_path):
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_partial_benchmark.py'
    spec = importlib.util.spec_from_file_location('prepare_partial_benchmark_missing_replay', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    membership = tmp_path / 'membership.json'
    membership.write_text(json.dumps({'rows': [{'obj_id': 'MISSING_FIXTURE', 'partition': 'design', 'order_key': 'a'}]}))
    out = tmp_path / 'out'
    assert module.main(['--membership', str(membership), '--n', '1', '--horizons', '5',
                        '--reuse-raw', str(tmp_path / 'missing-source'), '--out', str(out)]) == 1
    exported = [RecordedSession.model_validate_json(line) for line in (out / 'sessions.jsonl').read_text().splitlines()]
    assert len(exported) == 1
    assert [o.status for o in exported[0].observations] == ['error'] * 4 + ['unavailable']
    assert json.loads((out / 'receipt.json').read_text())['status_counts'] == {'error': 1}


def test_ignored_truth_channel_cannot_change_input_validation(fake_science):
    clean = events()
    with_text_label = clean.astype(object)
    with_text_label[:, 14] = 'IGNORED_TRUTH'
    first = collect_flare('FIXTURE', 5, events_array=clean, context={'host_sep': 1})
    second = collect_flare('FIXTURE', 5, events_array=with_text_label, context={'host_sep': 1})
    assert [o.status for o in first.observations] == [o.status for o in second.observations]
    assert [o.payload for o in first.observations] == [o.payload for o in second.observations]
    assert 'IGNORED_TRUTH' not in second.model_dump_json()
