"""Cohort identity is invariant to packaging and rejects scientific-row changes."""
import copy
import json
from pathlib import Path

import pytest
from applecider_agents.core import membership


def fixture_document():
    return {'sources': [{'path': 'splits/train.csv'}], 'rows': [
        {'obj_id': 'FIXTURE_B', 'partition': 'holdout', 'order_key': 'b', 'forced_design': False},
        {'obj_id': 'FIXTURE_A', 'partition': 'design', 'order_key': 'a', 'forced_design': True}]}


def test_digest_is_row_order_and_packaging_invariant():
    original = fixture_document()
    repackaged = copy.deepcopy(original)
    repackaged['sources'][0]['path'] = 'external-copy/splits/train.csv'
    repackaged['description'] = 'portable metadata packaging'
    repackaged['rows'].reverse()
    assert membership.canonical_row_digest(original) == membership.canonical_row_digest(repackaged)


@pytest.mark.parametrize('field,value', [('obj_id', 'CHANGED_ID'), ('partition', 'different'),
                                        ('order_key', 'changed'), ('forced_design', True)])
def test_frozen_loader_rejects_any_scientific_row_change(monkeypatch, tmp_path, field, value):
    original = fixture_document()
    monkeypatch.setattr(membership, 'FROZEN_MEMBERSHIP_ROW_SHA256', membership.canonical_row_digest(original))
    path = tmp_path / 'membership.json'
    path.write_text(json.dumps(original))
    loaded = membership.load_frozen_membership(path)
    assert loaded['rows'] == original['rows'] and loaded['raw_sha256']
    changed = copy.deepcopy(original)
    changed['rows'][0][field] = value
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='frozen cohort'):
        membership.load_frozen_membership(path)


def test_loader_accepts_portable_metadata_with_exact_frozen_rows(monkeypatch, tmp_path):
    original = fixture_document()
    digest = membership.canonical_row_digest(original)
    monkeypatch.setattr(membership, 'FROZEN_MEMBERSHIP_ROW_SHA256', digest)
    portable = copy.deepcopy(original)
    portable['source_repository'] = 'https://example.invalid/scientific-dataset'
    portable['sources'][0]['path'] = 'portable/splits/train.csv'
    portable['rows'].reverse()
    portable['semantic_row_sha256'] = digest
    path = tmp_path / 'public.json'
    path.write_text(json.dumps(portable))
    assert membership.load_frozen_membership(path)['semantic_row_sha256'] == digest


def test_canonical_rows_reject_duplicate_or_missing_forced_assignment():
    document = fixture_document()
    document['rows'].append(dict(document['rows'][0], partition='another'))
    with pytest.raises(ValueError, match='duplicate'):
        membership.canonical_row_digest(document)
    document = fixture_document()
    del document['rows'][0]['forced_design']
    with pytest.raises(ValueError, match='boolean'):
        membership.canonical_row_digest(document)


def test_released_public_membership_matches_frozen_scientific_identity():
    path = Path(__file__).resolve().parents[1] / 'benchmarks/evidence_policy_membership.json'
    document = membership.load_frozen_membership(path)
    assert len(document['rows']) == 8922
    assert document['semantic_row_sha256'] == membership.FROZEN_MEMBERSHIP_ROW_SHA256
    assert all(not Path(source['path']).is_absolute() for source in document['sources'])
    text = path.read_text()
    assert '/workspace/' not in text and '/tmp/' not in text and '.aris/' not in text
