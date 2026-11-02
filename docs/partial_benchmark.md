# Partial-policy acquisition benchmark

This benchmark evaluates faithful execution of a frozen software policy on
captured tool observations. It measures decision certification, requested report
completion and imposed acquisition costs. It does not measure astronomical
accuracy, follow-up utility or classifier generalization. EC2 and stochastic
function evaluation are established methods.

Use [the observation interface](partial_observations.md) to capture typed
`RecordedSession` files. The runner also accepts raw captures. Snapshot filenames
are `<obj_id>_h<horizon>.json`, with relative horizons 5, 10, 20 and 100 days.
`benchmarks/evidence_policy_membership.json` contains the frozen scientific cohort;
metadata packaging can change only if the exact scientific-row digest is retained.
All horizons of an object stay in one partition. These public arrays overlap
classifier training/calibration; the holdout concerns policy execution.

```bash
python scripts/run_partial_benchmark.py \
  --snapshots /path/to/design-snapshots \
  --membership benchmarks/evidence_policy_membership.json \
  --output /path/to/new-results --partition design --limit 96 \
  --methods eager,compiled,fail_closed,ec2 \
  --profiles decision_only,legacy_report \
  --cost-scenarios shared --budgets 40 --scenarios clean
```

`decision_only` requires a certified policy verdict/priority. `legacy_report` also
requires measured classify/anomaly diagnostics, consulted host context, and a
successful terminal history result when the frozen trigger is true. A decision
with missing report fields is an unfinished full-report task. Unknown, failed,
malformed or mismatched observations do not become observed negatives. Existing
explicit context Boolean flags remain the legacy policy inputs; missing upstream
raw measurements are disclosed rather than claimed repaired.

The logical domain is a conservative independent Boolean product. It retains all
completions consistent with individually valid predicates, independently of
scheduler weights. Candidate sets therefore overapproximate realizable scientific
states; singleton certification does not establish maximal coverage or physical
correctness. The ordinary legacy `Report` and opt-in partial result remain distinct.

The equal-policy controls share evidence validity, domain and report obligations:

- `eager`: ordinary diagnostics and conditionally history, with unknown propagation.
- `compiled`: a fixed order and certificate check after each query. Tuning scores
  all 120 orders using actual runtime completion for all 32 acquired-tool subsets.
- `fail_closed`: direct nullable policy execution; an unavailable next branch
  condition causes abstention. Its query order can differ from compiled execution.
- `ec2`: expected cross-decision edge cutting per incremental cost. Child edge
  masses use global weights; outcome probabilities normalize separately. Known
  bits in partial payloads condition the posterior, unknown bits remain
  marginalized, and report-source validity remains a separate obligation.
- `exact_dp`: finite-prior bounded-budget probability of task completion first,
  expected spend second. Irreversible failures stay in its probability mass.
  State/time failures remain failed comparator requests. The clean prior models
  complete scientific payload outcomes plus unavailable history; partial/fault
  outcomes outside that prior are not claimed modeled optima.

`legacy_eager_prefix` is a partial default-behavior diagnostic, not an independent
legacy oracle. `scripts/run_legacy_diagnostic.py` separately executes the actual
pinned `mock_triage` AST with private replay tool facades. It retains the literal
Report, calls and exceptions. This is matched captured-prefix policy replay, not a
live replay of the old omitted-horizon/default-100 context pipeline. Unavailable
status becomes an explicit error envelope, never measured history or a negative.
Literal reports do not certify the new profile, and this unbudgeted diagnostic is
excluded from fair scheduler-gain comparisons.

Each logical call costs 1 imposed unit. Shared costs add FLARE 20 once for
classify/anomaly, local events 2 once for context/photometry, and history 8.
Independent costs add 20 for each model call, 2 each for context/photometry and 8
for history. Complete five-tool costs are 35 and 57. These are fixed stress
scenarios, not fitted live latencies. Successful shared-cache behavior is modeled;
real failures can retry or fail before inference. Actual source backend counters,
whole-session wall time and per-observation time retain their distinct scopes.
Snapshot replay establishes no live acquisition or inference savings.

Primary budget is 60 under both cost scenarios, with `legacy_report` and shared
costs primary. Secondary budgets 5, 25 and 40 are separate outcomes. Completed
blind reports require the same four scientific tools: the imposed floor is 26
shared or 48 independent units. Below these bounds, an analytical zero-query
abstention control has task completion 0 and acquisition cost 0. It is a static
feasibility result, not a measured scheduler or positive decision certificate.
Spending below an infeasible budget cannot support a gain claim against it.

The selected orders and exact tune prior are shipped as portable resources.
Always pass `--prior` explicitly: the JSON's adjacent relative `prior_path` is a
resource reference, and the current CLI does not resolve it relative to the file.

```bash
python scripts/run_partial_benchmark.py \
  --snapshots /path/to/tune-snapshots \
  --membership benchmarks/evidence_policy_membership.json \
  --frozen-config benchmarks/evidence_policy_selected_config.json \
  --prior benchmarks/evidence_policy_prior.npz \
  --output /path/to/new-tune-replay --partition tune --limit 256 \
  --profiles decision_only,legacy_report \
  --cost-scenarios shared,independent --budgets 60,5,25,40
```

This command performs comparative replay. Configuration selection is separate:
`--tune-config-out` selects orders and then also runs comparisons. A selection-only
caller must invoke `fit_tune_configuration` without the CLI replay stage.
Tuning used the first 256 tune objects at all horizons. Missing history receives
uniform latent modeling mass, never an empirical history label. Support-complete
scheduling combines 0.95 tune mass with 0.05 uniform logical mass. The empirical
support ablation removes smoothing without changing the logical certificate.
Neither prior is astronomical world probability. Spectral truth is never a
scheduler feature or training target.

Five single-tool masks and context+history/classify+anomaly pairs remove only
preexisting valid observations, with eligibility recorded. Three further classify
scope interventions alter object, horizon or session token. A stale-token mismatch
is distinct from actual cache reseeding, which has separate integration tests.
These are controlled interventions, not measured fault frequencies. The negative
semantics ablation deliberately treats unknown values as false and is audited
against actual evidence with the same compiled order, profile and cost.

Every request remains in `rows.jsonl`, including exceptions, unresolved decisions,
unavailable sources and incomplete reports. Summaries separate internal logical
certification from independent reference availability/error; `no_reference` is
never imputed. Paired intervals use 2,000 object-level resamples, seed 20261102.
Zero-error exact bounds use object events on referenced emitted looks; four
horizons are not four independent objects. Cost on a both-complete subset is
explicitly labeled. Missing cost receipts suppress the affected mean/interval.

A known-unavailable-history trigger could support an additional early-abstention
guard on impossible full reports. That all-request optimization is not implemented
or claimed as a new method here. It would need a matching conservative control,
separate decision/report coverage and actual-work accounting. No global scheduling
optimality or live-latency gain follows from the current results.

Confirmation requires a complete configuration/analysis freeze and a fresh source
review plus explicit release. [Current results](partial_benchmark_results.md)
contain completed design diagnostics and tune selection only. Confirmation capture
and partial replay were stopped at the user request; the checkpoint does not
provide a complete confirmation estimate.

Public analysis commands validate the full requested Cartesian grid before
summarizing it. Cross-support pairing is explicit; it does not group the two
priors into unrelated comparisons against compiled execution.

```bash
python scripts/analyze_partial_benchmark.py --mode merge \
  --inputs /path/to/four-main-shards/rows-1.jsonl /path/to/rows-2.jsonl \
    /path/to/rows-3.jsonl /path/to/rows-4.jsonl \
  --membership benchmarks/evidence_policy_membership.json \
  --partition confirm --limit 1024 --output /path/to/main-analysis.json
python scripts/analyze_partial_benchmark.py --mode support \
  --inputs /path/to/main-logical-rows.jsonl \
  --empirical-inputs /path/to/empirical-ec2-rows.jsonl \
  --membership benchmarks/evidence_policy_membership.json \
  --partition confirm --limit 1024 --output /path/to/prior-analysis.json
python scripts/analyze_partial_benchmark.py --mode ec2-trace \
  --inputs /path/to/main-logical-rows.jsonl \
  --empirical-inputs /path/to/empirical-ec2-rows.jsonl \
  --snapshots /path/to/captured-snapshots \
  --prior benchmarks/evidence_policy_prior.npz \
  --membership benchmarks/evidence_policy_membership.json \
  --partition confirm --limit 1024 --output /path/to/ec2-traces.json
```

The trace command reconstructs pre-query posterior regimes from recorded attempted
prefixes and verifies the selected actions. Empty/zero-weight posterior, zero
cross-class edges and positive edge-cut steps are separate. Failed requests with
unknown execution prefixes retain an unavailable reconstruction, not a fabricated
zero-query trace. Used prior/snapshot/source hashes are recorded.

Literal design diagnostics and their evidence audit are reproducible separately:

```bash
python scripts/run_legacy_diagnostic.py --snapshots /path/to/design-snapshots \
  --membership benchmarks/evidence_policy_membership.json --partition design \
  --scenarios clean,missing_classify,missing_anomaly,missing_context,missing_photometry,missing_history,missing_context_history,missing_classify_anomaly \
  --output /path/to/literal-results
python scripts/analyze_partial_benchmark.py --mode literal \
  --inputs /path/to/literal-results/rows.jsonl --snapshots /path/to/design-snapshots \
  --membership benchmarks/evidence_policy_membership.json --partition design --limit 96 \
  --scenarios clean,missing_classify,missing_anomaly,missing_context,missing_photometry,missing_history,missing_context_history,missing_classify_anomaly \
  --output /path/to/literal-evidence-audit.json
```

The literal CLI also accepts the declared tune/confirm partitions with their
frozen limits; the study's actual confirmation release is controlled separately.
Public replay tools require frozen scientific inputs/configuration where relevant,
without embedding a particular review service or orchestration environment.

The study reuses the FLARE models and data, and established acquisition and
uncertainty concepts:

| Primary source | Reused concept and boundary |
|---|---|
| [Sasli et al., FLARE (2026 preprint)](https://arxiv.org/abs/2610.30656) | Photometric classifier/anomaly tools and source data; this wrapper adds no classifier training or independent accuracy/utility evidence. |
| [Golovin, Krause & Ray, EC2 (NeurIPS 2010)](https://arxiv.org/abs/1010.3091) | Known decision-class singleton stopping and edge-cut acquisition; no new algorithm or approximation guarantee is claimed. |
| [Deshpande, Hellerstein & Kletenik, SBFE (SODA 2014)](https://doi.org/10.1137/1.9781611973402.107) | Known costly function certification; determining a value and fulfilling report obligations are separate tasks. |
| [Cost-optimal decision diagrams (2026 preprint)](https://arxiv.org/abs/2606.24672) | Existing cost-sensitive function evaluation with correlated finite outcomes. |
| [CAGE (2026 preprint)](https://arxiv.org/abs/2607.29190) | Existing policy certification under typed-return uncertainty. |
| [RSure-Agent (2026 preprint)](https://arxiv.org/abs/2610.04836) and [VeriPhy (2026 preprint)](https://arxiv.org/abs/2609.03153) | Evidence/status/unknown/provenance obligations predate this scoped adapter. |

The contribution evaluated here is a software-policy adapter and an empirical
null under explicit scientific report obligations. It establishes no new general
acquisition method, agent-reasoning gain or astronomical follow-up utility.
