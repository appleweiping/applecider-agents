# Recorded CPU reproduction environment

Install this source checkout for the partial interface and benchmark scripts. The legacy baseline is `789b7955dc1ede4a7d7ec1a74d8d571e3528eba9`; it is not the feature implementation revision. Use the checkout containing this document and the source fingerprints recorded with the scientific results.

The tested environment used CPython 3.12.14 on Linux x86_64 (glibc 2.41), CPU execution and one OpenMP/BLAS thread per process. PyTorch was not installed in this profile; no CUDA or CPU Torch wheel is required for these recorded LightGBM/LUNA inference and replay runs. Other Python, operating-system or GPU profiles have not been validated here.

| Component | Tested identity |
|---|---|
| FLARE code, models and public data | `applecider-ml/flare` at `227cbff9b3b11d6cc54f42248be7fcde0be88a15` |
| LUNA | `asasli/luna` at `6f2ca0322cfe060ab9d70f25847d65a8378ead48` (`luna-ood 0.4.1`) |
| LUNA source archive SHA256 | `29c5827db5ae17a815615a4eaa3b52a676341e01688570fd6817473ee9f33e54` |
| Numerical/model stack | NumPy 2.3.5, pandas 2.3.3, SciPy 1.17.1, scikit-learn 1.8.0, LightGBM 4.6.0, light-curve 0.13.1, PyArrow 23.0.1 |
| API/schema stack | Pydantic 2.14.2, Anthropic 2.3.0, MCP 2.3.0, jsonschema 4.26.0 |
| Plotting | Matplotlib 3.10.8 |

`requirements-reproduction.txt` records the installed runtime/replay dependency closure. Research-only training, optimization and explanation extras are excluded. Matplotlib is also declared by the pinned upstream LUNA package; the plotting script remains optional and adds no dependency to the app's runtime declarations. `benchmarks/evidence_policy_environment.json` records relative source/checkpoint/data filenames, sizes and checksums, plus the app source fingerprints.

These commands are a portable recipe derived from that working environment. They have **not** been rerun as a fresh upstream-download rebuild in a new environment. Existing source/runtime and installed-application-wheel checks used the verified installed dependency environment. Package-version records and source checksums are not a claim of a fully hermetic build.

## Install from this checkout

Use CPython 3.12.14 and a system OpenMP runtime (`libgomp.so.1`, commonly supplied by `libgomp1` on Debian/Ubuntu). From this repository:

```bash
python3.12 -m venv ../applecider-reproduction-venv
. ../applecider-reproduction-venv/bin/activate
python -m pip install -r requirements-reproduction.txt
python -m pip install --force-reinstall --no-deps .
python -m pip check
```

The explicit source installation supplies this checkout's partial command even if a published package of the same version was installed previously. FLARE is pinned without moving optional-extra URLs; LUNA is installed from its recorded commit archive with SHA256 verification. This recipe needs Git and access to the ordinary public package/source hosts. It requires no application LLM key or SkyPortal credentials.

Keep input data and generated observations outside the checkout:

```bash
git clone https://github.com/applecider-ml/flare.git ../flare-data
git -C ../flare-data checkout --detach 227cbff9b3b11d6cc54f42248be7fcde0be88a15
tar -xzf ../flare-data/benchmark/light_curves.tar.gz -C ../flare-data/benchmark
export APPLECIDER_DATA_DIR="$PWD/../flare-data/benchmark"
export APPLECIDER_EVENTS_DIRS="$PWD/../flare-data/benchmark/events"
export APPLECIDER_AGENTS_HOME="$PWD/../applecider-reproduction-cache"
export APPLECIDER_AGENTS_LOG="$PWD/../applecider-reproduction-logs"
export FLARE_MODE=dry-run
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/check_partial_membership.py --benchmark-dir ../flare-data/benchmark
python scripts/prepare_partial_benchmark.py \
  --membership benchmarks/evidence_policy_membership.json --partition design --n 96 \
  --horizons 5 10 20 100 --events-dir ../flare-data/benchmark/events \
  --data-dir ../flare-data/benchmark --out ../design-observations
```

The fixed public source/model fingerprints cover the full-curve classifier, phase-matched early classifiers, their conformal/anomaly/Venn-Abers calibrations, consensus bundles/manifests, and public light-curve/context inputs used by this profile. They do not ship model weights or light curves in this agent repository. Data overlap with the classifier's fitting/calibration population remains disclosed in the [cohort guide](../benchmarks/README.md).

For comparative replay, use the [benchmark protocol](partial_benchmark.md), the released selected configuration and its explicit prior path. For the opt-in callable/CLI/MCP path, use the [observation guide](partial_observations.md). Absolute event origins and static catalogue availability dates remain unknown; the provided offline history is unavailable. No live-history search, app LLM trial, telescope-utility evaluation or new classifier fitting is part of this reproduction profile.
