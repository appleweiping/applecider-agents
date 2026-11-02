#!/usr/bin/env python
"""Prepare offline typed observations from explicit object membership.

Select any explicitly declared named membership partition. Experimental split
freezes and execution authorization are external to this preparation tool.
Use --reuse-raw to convert existing genuine raw-tool payloads
without new model acquisition. Never read labels or query a remote service.
"""
from __future__ import annotations
import argparse
import os
import platform
import re
from importlib.metadata import version
from collections import Counter
import hashlib
import json
import math
import uuid
from pathlib import Path
import time

from applecider_agents.core.observations import load_raw_snapshot, RecordedSession
from applecider_agents.core.evidence import EvidenceScope, ToolObservation
from applecider_agents.flare.observations import collect_flare


def validate_membership(document):
    """Return only identity fields after validating every declared object assignment."""
    if not isinstance(document, dict) or not isinstance(document.get('rows'), list):
        raise ValueError('membership must be a JSON object containing a rows list')
    validated, owners = [], {}
    for index, row in enumerate(document['rows']):
        if not isinstance(row, dict):
            raise ValueError(f'membership row {index} must be an object')
        for field in ('obj_id', 'partition', 'order_key'):
            value = row.get(field)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f'membership row {index} requires a nonempty trimmed string {field}')
        obj_id, partition = row['obj_id'], row['partition']
        if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', obj_id) is None:
            raise ValueError(f'membership row {index} has an invalid object identifier')
        if obj_id in owners:
            if owners[obj_id] != partition:
                raise ValueError(f'object {obj_id} is assigned to multiple partitions')
            raise ValueError(f'duplicate object ID {obj_id}')
        owners[obj_id] = partition
        validated.append({key: row[key] for key in ('obj_id', 'partition', 'order_key')})
    return validated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--membership', type=Path, required=True, help='JSON rows with obj_id,partition,order_key only')
    parser.add_argument('--partition', default='design', help='named partition explicitly declared in membership rows')
    parser.add_argument('--n', type=int, default=96)
    parser.add_argument('--horizons', type=float, nargs='+', default=(5, 10, 20, 100))
    parser.add_argument('--events-dir', type=Path)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--reuse-raw', type=Path, help='Directory containing raw snapshots/')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error('--n must be positive')
    if any(not math.isfinite(h) or h < 0 for h in args.horizons) or len(set(args.horizons)) != len(args.horizons):
        parser.error('--horizons must be distinct finite nonnegative values')
    try:
        membership_rows = validate_membership(json.loads(args.membership.read_text()))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(f'invalid membership: {exc}')
    if args.partition not in {row['partition'] for row in membership_rows}:
        parser.error(f'partition {args.partition!r} is not declared in membership')
    rows = sorted((r for r in membership_rows if r['partition'] == args.partition), key=lambda r: r['order_key'])[:args.n]
    if len(rows) != args.n:
        parser.error('requested object count exceeds partition membership')
    # Copy only approved identity fields; no truth or arbitrary source metadata.
    ids = [r['obj_id'] for r in rows]
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'snapshots').mkdir()
    started = time.perf_counter()
    from applecider_agents.core import data as agent_data
    from applecider_agents.flare import tools as agent_tools
    from flare import data as flare_data, skyportal as flare_service, features as flare_features
    source_files = {'prepare_script': Path(__file__), 'agent_observations': Path(collect_flare.__code__.co_filename),
        'agent_data': Path(agent_data.__file__), 'agent_tools': Path(agent_tools.__file__),
        'flare_data': Path(flare_data.__file__), 'flare_service': Path(flare_service.__file__),
        'flare_features': Path(flare_features.__file__)}
    manifest = {'schema': 'RecordedSession-v1', 'partition': args.partition, 'object_ids': ids,
        'horizons': args.horizons, 'planned_sessions': len(ids) * len(args.horizons),
        'membership_sha256': hashlib.sha256(args.membership.read_bytes()).hexdigest(),
        'source_file_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_files.items()},
        'package_versions': {name: version(name) for name in ('applecider-agents', 'flare-transients')},
        'resources': {'process_workers': 1, 'python_version': platform.python_version(),
            'thread_settings': {name: os.environ.get(name) for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}},
        'source_mode': 'saved_payload_conversion' if args.reuse_raw else 'actual_local_acquisition',
        'blind': True, 'absolute_epoch_known': False, 'static_catalogue_availability_known': False}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    counts = Counter()
    with (args.out / 'denominator.jsonl').open('w') as ledger, (args.out / 'sessions.jsonl').open('w') as sessions:
        for oid in ids:
            for horizon in args.horizons:
                ledger.write(json.dumps({'obj_id': oid, 'horizon_days': horizon, 'status': 'planned'}) + '\n')
                ledger.flush()
                tick = time.perf_counter()
                try:
                    if args.reuse_raw:
                        session = load_raw_snapshot(args.reuse_raw / 'snapshots' / f'{oid}_h{horizon:g}.json')
                    else:
                        session = collect_flare(oid, horizon, event_dirs=[args.events_dir] if args.events_dir else None,
                                                data_dir=args.data_dir)
                    # Scope must agree with membership/request before exporting for policy use.
                    if session.scope.obj_id != oid or session.scope.horizon_days != horizon:
                        raise ValueError('snapshot identity does not match frozen request')
                    output = args.out / 'snapshots' / f'{oid}_h{horizon:g}.json'
                    session.save(output)
                    sessions.write(session.model_dump_json() + '\n')
                    statuses = Counter(o.status for o in session.observations)
                    status = 'captured_with_error' if statuses['error'] else 'captured'
                    row = {'obj_id': oid, 'horizon_days': horizon, 'status': status,
                           'observation_status_counts': dict(statuses), 'snapshot': f'snapshots/{output.name}'}
                except Exception as exc:
                    status = 'error'
                    reason = f'{type(exc).__name__}: {exc}'
                    scope = EvidenceScope(obj_id=oid, horizon_days=horizon, session_id=uuid.uuid4().hex)
                    failed = RecordedSession(scope=scope, observations=tuple(
                        ToolObservation(tool=tool, scope=scope, inputs={'obj_id': oid, 'horizon_days': horizon},
                            status='error' if tool != 'light_curve_history' else 'unavailable',
                            payload=None, provenance={'source': 'preparation_failure'}, availability={'reason': reason})
                        for tool in ('flare_classify', 'flare_anomaly', 'get_photometry', 'get_context', 'light_curve_history')),
                        metadata={'blind': True, 'preparation_failed': True}, costs={'kind': 'no_scientific_acquisition'})
                    output = args.out / 'snapshots' / f'{oid}_h{horizon:g}.json'
                    failed.save(output)
                    sessions.write(failed.model_dump_json() + '\n')
                    row = {'obj_id': oid, 'horizon_days': horizon, 'status': status,
                           'error': f'{type(exc).__name__}: {exc}'}
                counts[status] += 1
                row['preparation_wall_seconds'] = time.perf_counter() - tick
                ledger.write(json.dumps(row) + '\n')
                ledger.flush()
                sessions.flush()
    receipt = {'planned_sessions': manifest['planned_sessions'], 'terminal_records': sum(counts.values()),
        'status_counts': dict(counts), 'preparation_wall_seconds': time.perf_counter() - started,
        'timing_interpretation': 'conversion overhead only' if args.reuse_raw else 'actual local observation acquisition',
        'scientific_quality_claim': False}
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    return int(counts['error'] > 0 or counts['captured_with_error'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
