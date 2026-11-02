#!/usr/bin/env python
"""Check the released scientific cohort and optionally its external CSV provenance.

Only membership metadata and opaque file checksums are read; no model calls or
scientific outcomes are evaluated.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from applecider_agents.core.membership import load_frozen_membership


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--membership', type=Path,
        default=Path(__file__).resolve().parents[1] / 'benchmarks/evidence_policy_membership.json')
    parser.add_argument('--benchmark-dir', type=Path, help='external FLARE benchmark directory for optional CSV checksum verification')
    args = parser.parse_args(argv)
    document = load_frozen_membership(args.membership)
    checked = []
    if args.benchmark_dir is not None:
        for source in document['sources']:
            relative = Path(source['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('CSV source path must be relative to the benchmark directory')
            actual = hashlib.sha256((args.benchmark_dir / relative).read_bytes()).hexdigest()
            if actual != source['sha256']:
                raise ValueError(f'CSV source checksum mismatch: {relative}')
            checked.append(str(relative))
    print(json.dumps({'objects': len(document['rows']),
        'partitions': dict(Counter(row['partition'] for row in document['rows'])),
        'semantic_row_sha256': document['semantic_row_sha256'], 'raw_sha256': document['raw_sha256'],
        'source_csv_checksums_verified': checked, 'science_or_model_outcomes_read': False}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
