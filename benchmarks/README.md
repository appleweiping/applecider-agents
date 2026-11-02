# Evidence-policy object cohort

`evidence_policy_membership.json` provides the frozen 8,922-object cohort from the public FLARE train/validation lists. It contains object identifiers and split metadata, not light curves, spectral labels or model outcomes. The partitions contain 5,399 design, 1,755 tune and 1,768 confirmation objects; all horizons of an object retain one assignment.

The source CSV paths are relative to an external FLARE `benchmark/` directory. The JSON records their checksums, repository revision, partition salt/rule, forced-design assignments and classifier-training overlap. This is an agent-policy holdout; these objects overlap the shipped classifier's training/calibration and do not provide independent classifier validation.

Scientific membership identity covers exactly `obj_id`, `partition`, `order_key` and `forced_design`. Project those four fields, sort rows by `obj_id`, serialize compact JSON with sorted object keys and ASCII escapes, encode UTF-8, and compute SHA256. Row order and metadata/path packaging do not affect that digest. Every repeated object ID is rejected.

| Identity | SHA256 |
|---|---|
| Scientific rows | `3acd8ae697288bbfcde71a0172ce25b92eafd692bb15def9b6fa96c2f78dadea` |
| Original frozen file | `f066a102a102a551ff9b57ce382d3e3755e3575073a63ebe089291450412d5a0` |
| Portable JSON file | `380a56a61ef95b5c38ee80aaf05abe4d84abdc42456cc742780652876a494884` |

Check the released membership independently of its packaging:

```bash
python scripts/check_partial_membership.py
```

Optionally verify the opaque source CSV checksums against a local public FLARE data checkout:

```bash
python scripts/check_partial_membership.py --benchmark-dir flare-data/benchmark
```

Prepare a declared subset using this public file:

```bash
python scripts/prepare_partial_benchmark.py \
  --membership benchmarks/evidence_policy_membership.json \
  --partition design --n 96 --horizons 5 10 20 100 \
  --events-dir flare-data/benchmark/events --data-dir flare-data/benchmark \
  --out design-observations
```

The preparation interface accepts any declared partition. A study's protocol determines when its tuning or confirmation partition may be acquired and evaluated. Checking membership alone opens no scientific outcomes and authorizes no experiment.
