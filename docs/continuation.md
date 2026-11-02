# Continuation checkpoint

Work paused on 2026-11-02 at the user's quota boundary. The study is unfinished; this checkout and its accompanying scientific checkpoint are intended for continuation, not a final confirmation claim.

## Completed and pending

- The opt-in partial-evidence runtime, scoped observations, compiled/nullable/EC2 controls and public replay/analysis scripts are implemented. The latest full regression run passed **176 tests**. Installed-wheel checks matched all 22 package Python files.
- Design experiments and tune selection are complete. Selection used 256 tune objects, four horizons and all 120 orders in each of 16 cells. No configuration was fitted on confirmation data.
- All **4,096 confirmation captures** exist: 1,024 objects × horizons 5, 10, 20 and 100. They contain 16,384 successful local tool observations and 4,096 explicitly unavailable histories, with no network/history queries or application-LLM trials.
- Confirmation replay is incomplete. An infrastructure interruption left 16,029 rows in the first attempt. An unchanged-code retry was then stopped normally at the user's request, preserving **129,037 complete rows**. These are overlapping attempts, not additional samples. The prior-support ablation, confirmation literal-policy diagnostic, full-grid analysis, final figure and final result audit are pending.
- A known delivery fix remains: `finite_number(10**400)` raises `OverflowError`. Unrepresentable numeric evidence should be rejected as unknown/invalid. Fix and test this separately after preserving the executed benchmark version; do not silently change frozen study code or recalculate selected orders.

The clean design comparison demonstrated no full-report acquisition advantage: all four controls completed 255/384 reports at 26 imposed shared-cost units each. This validates software-policy behavior, not astronomical utility or a new algorithm. Confirmation must not be described as complete from its partial files.

## Frozen scientific configuration

The primary task remains `legacy_report`, shared costs, budget 60. The complete clean grid is four methods × two profiles × two cost scenarios × budgets 60, 5, 25, 40: **262,144 requested rows**. The additional empirical-prior EC2 ablation has 8,192 rows; the clean literal-policy diagnostic has 4,096. Do not expand the cohort, retune, select favorable cells or weaken report obligations.

| Artifact | SHA256 |
|---|---|
| `benchmarks/evidence_policy_selected_config.json` | `09e6801a2e1a43901e60973497750bf1b9fc56902cffd049734dc876efccc101` |
| `benchmarks/evidence_policy_prior.npz` | `1d5cc201370f88c4b9cb6a8470d47f4ec0645496ceaeb92016956211ca53f822` |
| `benchmarks/evidence_policy_membership.json` | `380a56a61ef95b5c38ee80aaf05abe4d84abdc42456cc742780652876a494884` |
| `scripts/run_partial_benchmark.py` | `e534d0aef6407021929991eb315482d8f6b61bc50106422dbfacb9359f8a4942` |
| `scripts/analyze_partial_benchmark.py` | `fa8cc5425e0d5115597363a4c578b542f9f15de8215896ab2cb0b0aae9bc9362` |
| `scripts/run_legacy_diagnostic.py` | `917daf9ac213621de14852a371c94c031e6775469d286928aa34676b725fa83c` |

The cohort semantic digest is `3acd8ae697288bbfcde71a0172ce25b92eafd692bb15def9b6fa96c2f78dadea`. Configuration scientific digest is `f350c464de6e34f757d639521d5ee032bbcd4160b72e4d88257060c25f4f72be`. The main prior is 0.95 fitted tune mass plus 0.05 uniform logical mass; certificate admissibility remains independent of the prior.

## Environment and replay

Install **this source checkout**, not an older published package. Follow [the pinned environment recipe](reproduction_environment.md), [benchmark instructions](partial_benchmark.md), and the accompanying checkpoint inventory. The tested upstream revisions are FLARE `227cbff9b3b11d6cc54f42248be7fcde0be88a15` and LUNA `6f2ca0322cfe060ab9d70f25847d65a8378ead48`, using CPython 3.12.14 on Linux x86_64. The tested inference stack uses LightGBM/LUNA; Torch was not installed. A fresh remote-download rebuild has not been independently validated.

Restore the existing confirmation captures from the scientific checkpoint and verify its checksums. Do not reacquire them merely because controller replay was interrupted. If the checkpoint is unavailable, the public preparer can regenerate captures from the pinned external data; that would be a separately recorded acquisition.

The following commands reproduce the unchanged complete grid into **fresh** output directories. They do not append to or silently complete the retained partial attempts. Before further execution, record whether continuation fills missing keys or performs another uniform full retry, preserve every attempt, and verify all overlapping scientific fields exactly; exclude only `replay_controller_wall_seconds`. Never double the object/sample denominator. Keep at most four single-thread workers and the existing eight-CPU-hour study cap, including prior attempts and explicitly uncertain overhead.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export STUDY_SNAPSHOTS=/path/to/restored/confirmation/snapshots
for study_profile in decision_only legacy_report; do
  for study_cost in shared independent; do
    python scripts/run_partial_benchmark.py \
      --snapshots "$STUDY_SNAPSHOTS" \
      --membership benchmarks/evidence_policy_membership.json \
      --frozen-config benchmarks/evidence_policy_selected_config.json \
      --prior benchmarks/evidence_policy_prior.npz \
      --partition confirm --limit 1024 \
      --methods eager,compiled,fail_closed,ec2 \
      --profiles "$study_profile" --cost-scenarios "$study_cost" \
      --budgets 60,5,25,40 --scenarios clean \
      --support logical --missing-semantics unknown \
      --output "runs/confirmation-${study_profile}-${study_cost}"
  done
done
```

Run the same benchmark CLI for the prior ablation with `--methods ec2 --profiles decision_only,legacy_report --cost-scenarios shared --budgets 60 --support empirical`, retaining all other scientific arguments and using a separate fresh output directory. Run `scripts/run_legacy_diagnostic.py` with the same snapshots/membership, `--partition confirm --limit 1024 --scenarios clean` and a separate output directory. Its policy replay requires the Git history containing base revision `789b7955dc1ede4a7d7ec1a74d8d571e3528eba9`.

Then use `scripts/analyze_partial_benchmark.py` modes `merge`, `support`, `ec2-trace` and `literal`, as documented in the benchmark instructions. Validate complete unique grids before reporting results. Cross-support comparisons must pair the same objects/horizons; intervals use 2,000 object-cluster bootstrap replicates, seed 20261102. Do not pass partial files as a completed grid.

## Interpretation and remaining delivery work

Report budgets below 26 shared / 48 independent are infeasible under the declared cold-request tool/profile/cost model; a no-query unresolved control costs zero there. Spending differences in those cells are not certificate-efficiency gains. The bound is not a universal physical optimum. The Boolean domain overapproximates realizable astronomical states. Classifier training/calibration overlap, relative-prefix rather than absolute historical timing, upstream synthesized context flags and unavailable history remain explicit limitations.

Preserve the interrupted-attempt records and uncertainty: the first interruption has 353.38 observed aggregate child CPU seconds, without complete terminal resource receipts. The normally stopped retry measured 1,749.756072 child CPU seconds. These are scoped controller resources, separate from imposed acquisition costs. No uninterrupted end-to-end speedup is established.

After completing the fixed grid and diagnostics: verify overlap and denominators, perform the independent final result audit, export the complete public aggregate, render the all-cell figure, finish the bounded numeric-validation fix with unchanged-valid-input checks, and rebuild/reverify the delivered package if runtime source changes. The accompanying archive is a checkpoint; its presence does not imply that a final study or public archival release has been completed.
