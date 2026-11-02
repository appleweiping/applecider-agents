# Local partial-evidence observations

The opt-in path captures local FLARE tool results and evaluates the versioned software policy with missing evidence left unknown. It returns possible verdict/priority pairs and separate `decision_certified` and `report_complete` fields. It does not publish a SkyPortal comment or create a legacy `Report`.

```bash
applecider-agents partial ZTF_OBJECT_ID --horizon 5 --profile legacy_report
applecider-agents partial --snapshot saved-session.json --profile decision_only
```

Configure local event directories with `APPLECIDER_EVENTS_DIRS` and the scientific context table directory with `APPLECIDER_DATA_DIR`. The new path does not fetch photometry, history, catalogue classifications or an LLM response. Missing local inputs produce recorded errors; full history is explicitly unavailable. A saved session uses `RecordedSession` JSON, including its object, relative horizon and session identity. `--save-snapshot new-file.json` preserves the observations for replay. MCP exposes the same local opt-in path as `flare_partial_triage`.

```python
from applecider_agents.flare.observations import collect_flare
from applecider_agents.flare.partial import evaluate_flare

session = collect_flare("ZTF_OBJECT_ID", 5)
result = evaluate_flare(session.scope, session.observations,
                        profile="legacy_report", blind=True)
session.save("new-session.json")
```

Callers can supply `events_array` and `context` directly. Each call owns a copied data snapshot and local science caches, so concurrent requests for the same object cannot replace one another's data. The classification and anomaly tools share one inference inside a session. A new call gets a new session identity by default; applications supplying their own `session_id` must change it when replacing effective inputs.

The horizon is relative time since the supplied event origin. Absolute epochs and static catalogue availability dates are unknown. Peak-derived context uses the same observed prefix; valid timestamps are required to select that prefix. Original cached color channels are retained as archival input, while FLARE recomputes colors from the prefix for science features. Truth columns and archive label metadata are excluded from exported science and acquisition-policy inputs. Missing raw context values are listed in metadata; the successful upstream tool's explicit boolean flags are preserved for policy fidelity. Those flags do not certify that an absent raw catalogue value was measured negative.

`get_context` also accepts `horizon_days`; its default remains100 days, preserving existing legacy triage calls. Legacy rule and posting defaults are unchanged.

## Preparing a reproducible observation dataset

The benchmark belongs to the [public FLARE repository](https://github.com/applecider-ml/flare/blob/227cbff9b3b11d6cc54f42248be7fcde0be88a15/benchmark/DATASET.md); it is not bundled with this agent checkout. With a FLARE data checkout at `flare-data`, unpack its published archive before preparation:

```bash
tar -xzf flare-data/benchmark/light_curves.tar.gz -C flare-data/benchmark
```

Use scientific inputs and model artifacts from a recorded compatible FLARE revision.

The released [scientific membership](../benchmarks/README.md) is available at `benchmarks/evidence_policy_membership.json`. Verify it with `python scripts/check_partial_membership.py`; the checker uses scientific-row identity, so portable metadata packaging does not change the cohort.

Supply a membership JSON containing `rows` with nonempty string `obj_id`, `partition`, and `order_key`. `--partition` selects any explicitly declared named partition; it defaults to `design`. The tool validates the entire manifest before selecting objects: object IDs must be unique globally, including across partitions, and identifiers must begin with a letter or digit and contain only ASCII letters, digits, underscores, dots or dashes. Duplicate rows and conflicting assignments are rejected. All horizons of a selected object stay in its declared partition. Experimental split freezes and decisions about when to acquire a partition belong to the research protocol, not hardcoded application role names.

```bash
python scripts/prepare_partial_benchmark.py \
  --membership membership.json --partition design --n 1 \
  --horizons 5 10 20 100 --events-dir flare-data/benchmark/events \
  --data-dir flare-data/benchmark --out observation-data
```

Outputs include `manifest.json`, typed `sessions.jsonl`, per-session snapshots, a planned/terminal `denominator.jsonl`, and `receipt.json`. Failed and unavailable inputs remain in these records. A nonzero exit reports an acquisition error without discarding failed sessions.

Existing genuine raw-tool exports can be converted using `--reuse-raw path-to-export`. `core.observations.load_raw_snapshot` retains recorded acquisition costs as provenance. Conversion time and cached replay do not establish live acquisition latency savings. The optional legacy100 context adapter retains its100-day scope, making an early-scope mismatch explicit.

The raw snapshot adapter accepts a JSON object with `obj_id`, `horizon_days`, optional truth-free `scientific_input` and acquisition `wall_seconds`, and an `observations` list. Each observation records `tool`, `input`, `status` (`success`, `error`, `unavailable`, or `not_required`), and its actual dictionary `result` or null. Optional `effective_photometric_horizon_days` retains scope; provenance can include `source`, `backend_work_delta`, `backend_records`, `scientific_input_ref`, and `event_file_sha256`. Canonical `get_context` is accepted; separately recorded `get_context_prefix` and `get_context_legacy100` variants can also be selected explicitly. Archive labels, spectral truth and membership labels do not belong in scientific inputs or policy features.

These artifacts support retrospective software-policy evaluation. They do not establish optimal astronomical actions, independent classifier validation, historical as-of safety, scheduler gains or scientific novelty.
