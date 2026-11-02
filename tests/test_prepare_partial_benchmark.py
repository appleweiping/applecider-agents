"""Preparation interface validation on fixtures only; no real data or science calls."""
import importlib.util
import json
from pathlib import Path

import pytest

from applecider_agents.core.evidence import EvidenceScope, ToolObservation
from applecider_agents.core.observations import RecordedSession


@pytest.fixture
def prepare_module():
    path = Path(__file__).resolve().parents[1] / 'scripts/prepare_partial_benchmark.py'
    spec = importlib.util.spec_from_file_location('prepare_membership_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(obj_id='FIXTURE_A', partition='design', order_key='a'):
    return {'obj_id': obj_id, 'partition': partition, 'order_key': order_key}


def test_arbitrary_declared_partition_uses_same_portable_interface(prepare_module, monkeypatch, tmp_path):
    requested = []
    def unavailable_fixture(obj_id, horizon, **kwargs):
        requested.append((obj_id, horizon))
        scope = EvidenceScope(obj_id=obj_id, horizon_days=horizon, session_id='fixture-snapshot')
        observation = ToolObservation(tool='flare_classify', scope=scope,
            inputs={'obj_id': obj_id, 'horizon_days': horizon}, status='unavailable', payload=None)
        return RecordedSession(scope=scope, observations=(observation,))
    monkeypatch.setattr(prepare_module, 'collect_flare', unavailable_fixture)
    path = tmp_path / 'membership.json'
    partition = 'validation epoch 2'
    path.write_text(json.dumps({'rows': [row('FIXTURE_B', partition, 'b'), row('FIXTURE_A', partition, 'a'), row('FIXTURE_OTHER', 'other')]}))
    output = tmp_path / 'prepared'
    assert prepare_module.main(['--membership', str(path), '--partition', partition, '--n', '1',
        '--horizons', '5', '--out', str(output)]) == 0
    assert requested == [('FIXTURE_A', 5)]
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['partition'] == partition and manifest['object_ids'] == ['FIXTURE_A']
    assert json.loads((output / 'receipt.json').read_text())['terminal_records'] == 1


@pytest.mark.parametrize('document', [
    [], {'rows': 'not-a-list'}, {'rows': [None]}, {'rows': [{}]},
    {'rows': [row(obj_id='')]}, {'rows': [row(obj_id='../not-an-object')]},
    {'rows': [row(obj_id=123)]}, {'rows': [row(partition='')]},
    {'rows': [row(order_key=123)]}, {'rows': [row(order_key=' ')]},
    {'rows': [row(), row()]},
    {'rows': [row(), row(partition='unselected_partition')]},
    {'rows': [row(), row('FIXTURE_OTHER', 'unselected_partition', '')]},
])
def test_whole_manifest_rejected_before_output_or_acquisition(prepare_module, monkeypatch, tmp_path, document):
    monkeypatch.setattr(prepare_module, 'collect_flare', lambda *a, **kw: pytest.fail('invalid membership reached acquisition'))
    path = tmp_path / 'membership.json'
    path.write_text(json.dumps(document))
    output = tmp_path / 'prepared'
    with pytest.raises(SystemExit) as exc:
        prepare_module.main(['--membership', str(path), '--n', '1', '--out', str(output)])
    assert exc.value.code == 2
    assert not output.exists()


def test_undeclared_partition_rejected_without_acquisition(prepare_module, monkeypatch, tmp_path):
    monkeypatch.setattr(prepare_module, 'collect_flare', lambda *a, **kw: pytest.fail('undeclared partition reached acquisition'))
    path = tmp_path / 'membership.json'
    path.write_text(json.dumps({'rows': [row()]}))
    output = tmp_path / 'prepared'
    with pytest.raises(SystemExit) as exc:
        prepare_module.main(['--membership', str(path), '--partition', 'absent', '--n', '1', '--out', str(output)])
    assert exc.value.code == 2 and not output.exists()
