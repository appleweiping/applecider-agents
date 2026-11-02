import importlib.util
from pathlib import Path
import sys

import pytest
from applecider_agents.core.evidence import EvidenceScope, ToolObservation
from applecider_agents.core.observations import RecordedSession

scripts = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('legacy_diagnostic', scripts / 'run_legacy_diagnostic.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def session(payloads):
    scope = EvidenceScope(obj_id='fixture', horizon_days=20, session_id='fixture')
    return RecordedSession(scope=scope, observations=tuple(ToolObservation(tool=tool, scope=scope,
        inputs={'obj_id': 'fixture', 'horizon_days': 20}, status='success', payload=payload)
        for tool, payload in payloads.items()))


def test_pinned_function_preserves_classify_error_fallback_and_calls():
    code, _ = diagnostic.pinned_function(Path(__file__).parents[1])
    result = diagnostic.replay(session({'flare_classify': {'error': 'controlled failure'}}), code)
    assert result['literal_report']['verdict'] == 'insufficient_data'
    assert result['literal_report']['energy_percentile'] == 0
    assert [c['tool'] for c in result['logical_calls']] == [
        'flare_classify', 'flare_anomaly', 'get_context', 'get_photometry']
    assert not result['new_profile_certified']


def test_pinned_function_exception_is_retained():
    code, _ = diagnostic.pinned_function(Path(__file__).parents[1])
    result = diagnostic.replay(session({'flare_classify': {'obj_id': 'fixture'}}), code)
    assert result['literal_report'] is None
    assert result['failure'].startswith('KeyError:')
    assert result['n_logical_calls'] == 4


def test_changed_source_is_rejected(tmp_path):
    # Use a real pinned source blob with a deliberately changed working copy.
    import subprocess
    path = tmp_path / diagnostic.TRIAGE_PATH
    path.parent.mkdir(parents=True)
    path.write_text('# changed')
    # Git source lookup is controlled by this fixture; no science is evaluated.
    original = diagnostic.subprocess.check_output
    diagnostic.subprocess.check_output = lambda *a, **k: b'def mock_triage(): pass\n'
    try:
        with pytest.raises(ValueError, match='differs from pinned'):
            diagnostic.pinned_function(tmp_path)
    finally:
        diagnostic.subprocess.check_output = original
