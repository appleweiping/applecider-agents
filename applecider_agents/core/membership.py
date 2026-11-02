"""Portable identity for the fixed scientific object cohort, independent of paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROW_FIELDS = ('obj_id', 'partition', 'order_key', 'forced_design')
FROZEN_MEMBERSHIP_ROW_SHA256 = '3acd8ae697288bbfcde71a0172ce25b92eafd692bb15def9b6fa96c2f78dadea'


def canonical_row_digest(document: dict) -> str:
    """Digest exact scientific rows; metadata/path changes do not change the cohort.

    Project the four ROW_FIELDS, sort rows by obj_id, then encode compact JSON
    with sorted object keys, ensure_ascii=True and UTF-8. No label data is used.
    """
    if not isinstance(document, dict) or not isinstance(document.get('rows'), list):
        raise ValueError('membership requires a rows list')
    rows, seen = [], set()
    for row in document['rows']:
        if not isinstance(row, dict):
            raise ValueError('membership rows must be objects')
        for field in ROW_FIELDS[:3]:
            value = row.get(field)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f'membership requires a nonempty trimmed string {field}')
        if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', row['obj_id']) is None:
            raise ValueError('invalid membership object identifier')
        if type(row.get('forced_design')) is not bool:
            raise ValueError('forced_design must be a boolean')
        if row['obj_id'] in seen:
            raise ValueError(f'duplicate membership object ID {row["obj_id"]}')
        seen.add(row['obj_id'])
        rows.append({field: row[field] for field in ROW_FIELDS})
    encoded = json.dumps(sorted(rows, key=lambda row: row['obj_id']), sort_keys=True,
                         separators=(',', ':'), ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def load_frozen_membership(path: str | Path) -> dict:
    """Load the frozen cohort with portable metadata; any scientific-row change fails."""
    raw = Path(path).read_bytes()
    document = json.loads(raw)
    actual = canonical_row_digest(document)
    if actual != FROZEN_MEMBERSHIP_ROW_SHA256:
        raise ValueError('membership scientific rows do not match the frozen cohort')
    if document.get('semantic_row_sha256', actual) != actual:
        raise ValueError('declared membership row digest is inconsistent')
    return {**document, 'semantic_row_sha256': actual, 'raw_sha256': hashlib.sha256(raw).hexdigest()}
