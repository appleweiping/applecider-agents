# Agents for AppleCiDEr

Triage agents for the [AppleCiDEr](https://github.com/applecider-ml) transient models. The first module wraps
[FLARE](https://github.com/applecider-ml/flare) (photometric classification, conformal sets, blind consensus anomaly alarm);
ASTRANet (spectra) is the next folder.

Agents are kept separate from the model packages on purpose. A model repo stays a dependency-light,
citable classifier. This repo holds what sits between a model and a human: the deterministic tools that
fetch and compute, the typed report, the rules for reading the numbers, the LLM loop that composes a
verdict, one MCP server that exposes all of it to SkyPortal's assistant, and the gate that decides what
gets posted where. **Every number in a report comes from a tool call; the language model never does science.**

```
applecider_agents/
  core/      report.py (typed Report), object_tools.py (photometry, context, full history, BTS, TNS),
             loop.py (Claude tool-use loop), mcp.py (one server, all models), skyportal_client.py, eval.py, registry.py
  flare/     tools.py (flare_classify, flare_anomaly, flare_analyze_photometry), triage.py (reading guide + rule verdict)
  astranet/  (next) get_spectrum via SkyPortal, astranet_classify, conformal set, spectral anomaly
```

## Install and run

```bash
pip install "applecider-agents[flare,mcp]"      # flare-transients[skyportal] + the MCP SDK
applecider-agents triage ZTF22aazcdan            # rule-based verdict, no LLM, works anywhere
applecider-agents triage ZTF22aazcdan --llm      # Claude loop over the same tools (ANTHROPIC_API_KEY)
applecider-agents mcp --transport streamable-http --port 8765 --models flare
applecider-agents eval --model flare --list ood  # the benchmark's out-of-taxonomy hold-out, run blind (no catalogue tools)
```

The benchmark tables (catalogue and context blocks) are read from the public flare repository and cached
locally on first use; light curves come from local event files when `APPLECIDER_EVENTS_DIRS` is set and
otherwise from the broker archive through `flare.fetch` (BOOM when configured). No path is machine-specific,
so the same commands run in a container or next to SkyPortal.

## Partial-evidence triage (opt-in)

Install this source checkout using the [recorded CPU reproduction profile](docs/reproduction_environment.md)
for the opt-in interface:

For local or saved FLARE observations, the `partial` command keeps missing or failed evidence unknown
and returns possible verdict/priority pairs, with separate `decision_certified` and `report_complete`
fields. It uses no LLM, live history lookup or publishing. Configure local inputs with
`APPLECIDER_EVENTS_DIRS` and `APPLECIDER_DATA_DIR`:

```bash
python -m pip install -r requirements-reproduction.txt
python -m pip install --force-reinstall --no-deps .
applecider-agents partial ZTF_OBJECT_ID --horizon 5 --profile legacy_report
```

The Python API (`collect_flare`, `evaluate_flare`) and MCP tool `flare_partial_triage` expose the same
opt-in path. Certificates concern the frozen software policy within a conservative predicate domain;
they do not certify astronomical accuracy or free-text claims. Existing `triage` defaults are unchanged.

See the [API, CLI and observation guide](docs/partial_observations.md),
[reproduction protocol](docs/partial_benchmark.md), [public cohort](benchmarks/README.md), and
[current design/tuning findings](docs/partial_benchmark_results.md).
The [continuation checkpoint](docs/continuation.md) records the incomplete confirmation run and remaining work;
no final confirmation aggregate or figure is included.

## What the agent does in SkyPortal

FLARE already runs as an analysis service (OSG plugin) and writes a classification and an annotation set
on the source page. The agent adds the layer above the numbers:

1. **Reads the whole object, not one score.** Classifier probabilities and the 90% set from the model
   matched to the light curve's phase (day 5, 10, 20 or full curve), credibility, the blind consensus
   anomaly alarm at 1%, 5% and 10% false-alarm rates, host offset and photo-z, Gaia
   astrometry, WISE colour, the years-long activity span of the full photometric history, the BTS and TNS records. One call,
   one verdict: ordinary / needs_spectrum / anomaly_review / likely_stellar_or_agn /
   already_classified / insufficient_data, with a priority 0-3 and one fact per tool as evidence.
2. **Knows the pitfalls the paper measured.** "SLSN in the set" is a true SLSN 21% of the time,
   "argmax SLSN" 64%, a photo-z luminosity below -21 mag alone 1%; a nuclear object active for three
   years is an AGN, not a transient; the anomaly alarm of an early light curve is calibrated against known
   objects with a similar number of detections, and the 1% alarm counts only from 8 detections. The
   reading guide travels with every `flare_triage` result, so SkyPortal's own assistant applies it.
3. **Answers questions, not just alerts.** Through MCP the assistant can be asked "why did FLARE call
   this SN CC?", "is this worth a spectrum tonight?", "which of tonight's twelve candidates are most likely
   to be superluminous?", and answer with tool results it can cite.
4. **Posts only through a gate.** Comments and annotations go out only under `FLARE_MODE=comment`, to
   named groups, with a daily cap; follow-up requests are never opened by the model.

## Modes and cost

`FLARE_MODE=dry-run` (default) never posts. The rule-based verdict costs nothing; the LLM loop is 6-9 tool
calls per object with `claude-opus-5` (a few cents). Every run appends tool calls, token usage and the
report to `APPLECIDER_AGENTS_LOG/reports.jsonl`.

## Adding a model

A folder with `tools.py` (the model's `@beta_tool` functions), `triage.py` (a reading guide and a
rule-based verdict returning `core.report.Report`), and an `__init__.py` that builds and registers a
`core.registry.ModelAgent`. The object tools, the LLM loop, the MCP server, the eval harness and the
SkyPortal gate are shared.
