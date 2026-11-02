#!/usr/bin/env python
"""Plot all prespecified cells from the released aggregate confirmation results.

Matplotlib is an optional plotting dependency, installed separately. This module
never queries science tools or evaluates a policy. Absolute means are not given
paired-contrast confidence intervals; those belong in the companion results.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import hashlib
import math
from pathlib import Path

METHODS = ('eager', 'compiled', 'fail_closed', 'ec2')
BUDGETS = (5, 25, 40, 60)
ROWS = (('legacy_report', 'shared'), ('legacy_report', 'independent'),
        ('decision_only', 'shared'), ('decision_only', 'independent'))
REPORT_FLOORS = {'shared': 26, 'independent': 48}
STYLES = {
    'eager': ('Conservative eager', '#0072B2', 'o', '-'),
    'compiled': ('Compiled', '#D55E00', 's', '--'),
    'fail_closed': ('Nullable fail-closed', '#009E73', '^', '-.'),
    'ec2': ('EC2', '#CC79A7', 'D', ':'),
}


@dataclass(frozen=True)
class Cell:
    profile: str
    cost: str
    budget: int
    method: str
    n_requested: int
    n_objects: int
    tasks_complete: int
    mean_spend: float | None

    @property
    def completion_fraction(self):
        return self.tasks_complete / self.n_requested


def validate_cells(cells):
    """Reject missing/duplicate cells, denominator changes and false floor points."""
    expected = {(profile, cost, budget, method) for profile, cost in ROWS
                for budget in BUDGETS for method in METHODS}
    by_key = {}
    denominators = set()
    for cell in cells:
        key = (cell.profile, cell.cost, cell.budget, cell.method)
        if key not in expected or key in by_key:
            raise ValueError(f'unknown or duplicate aggregate cell: {key}')
        if not (type(cell.n_requested) is int and type(cell.n_objects) is int and
                cell.n_requested > 0 and cell.n_objects > 0 and cell.n_requested == 4 * cell.n_objects):
            raise ValueError('aggregate must retain all four requested horizons per object')
        if type(cell.tasks_complete) is not int or not 0 <= cell.tasks_complete <= cell.n_requested:
            raise ValueError('invalid completed-task denominator')
        if cell.mean_spend is not None and (not math.isfinite(cell.mean_spend) or
                not 0 <= cell.mean_spend <= cell.budget + 1e-9):
            raise ValueError('invalid all-request imposed mean spend')
        if cell.profile == 'legacy_report' and cell.budget < REPORT_FLOORS[cell.cost] and cell.tasks_complete:
            raise ValueError('completed report below the declared analytical cost floor')
        by_key[key] = cell
        denominators.add((cell.n_requested, cell.n_objects))
    if set(by_key) != expected or len(denominators) != 1:
        raise ValueError('plot requires exactly 16 profile/cost/budget cells with four controls and one common requested cohort')
    return by_key


def render_figure(cells, output: Path, *, results_sha256: str, source_label: str):
    """Render descriptive absolute levels; no interval is manufactured from contrasts."""
    by_key = validate_cells(cells)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError('Plotting requires optional matplotlib: pip install matplotlib') from exc

    settings = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
                'axes.labelsize': 9, 'svg.fonttype': 'none', 'svg.hashsalt': results_sha256,
                'axes.spines.top': False,
                'axes.spines.right': False, 'savefig.facecolor': 'white'}
    with plt.rc_context(settings):
        fig, axes = plt.subplots(4, 2, figsize=(12.6, 12.5), sharex=True,
                                 gridspec_kw={'hspace': .47, 'wspace': .22})
        fig.suptitle('Task completion and imposed acquisition spend', fontsize=15, y=.985)
        n_requested = next(iter(by_key.values())).n_requested
        n_objects = next(iter(by_key.values())).n_objects
        fig.text(.5, .957, f'Software policy replay · {n_objects:,} confirmation objects · {n_requested:,} requests in every cell',
                 ha='center', fontsize=10, color='#444444')
        for row, (profile, cost) in enumerate(ROWS):
            label = 'Full report' if profile == 'legacy_report' else 'Decision only'
            row_title = f'{label} · {cost} costs'
            if row == 0:
                row_title += '  [primary: budget 60]'
            left, right = axes[row]
            left.set_title(row_title, loc='left', fontweight='bold', pad=10)
            right.set_title('Mean spend over every requested input', loc='left', pad=10)
            for ax in (left, right):
                ax.set_xlim(0, 65)
                ax.set_xticks(BUDGETS)
                ax.tick_params(axis='x', labelbottom=True)
                ax.grid(axis='y', color='#dddddd', linewidth=.6, zorder=0)
                if profile == 'legacy_report':
                    floor = REPORT_FLOORS[cost]
                    ax.axvspan(0, floor, color='#eeeeee', zorder=0)
                    ax.axvline(floor, color='#999999', linestyle=':', linewidth=1)
                    ax.text(floor + .7, .96, f'floor {floor}', transform=ax.get_xaxis_transform(),
                            ha='left', va='top', fontsize=8, color='#666666')
                if row == 0:
                    ax.axvspan(58.6, 61.4, color='#f7e5ae', alpha=.65, zorder=0)
            for index, method in enumerate(METHODS):
                name, color, marker, line = STYLES[method]
                series = [by_key[(profile, cost, budget, method)] for budget in BUDGETS]
                style = dict(color=color, marker=marker, linestyle=line, linewidth=1.45,
                             markersize=7-index*.5, markerfacecolor='white', markeredgewidth=1.2,
                             label=name, zorder=3+index)
                left.plot(BUDGETS, [cell.completion_fraction for cell in series], **style)
                # Missing cost accounting remains a gap, never a fabricated zero.
                right.plot(BUDGETS, [float('nan') if cell.mean_spend is None else cell.mean_spend
                                    for cell in series], **style)
            left.set_ylim(-.035, 1.055)
            left.set_yticks((0, .25, .5, .75, 1))
            left.set_ylabel('Task completion fraction')
            right.set_ylim(-1.5, 64)
            right.set_ylabel('Mean imposed spend (units)')
            left.set_xlabel('Imposed budget (units)')
            right.set_xlabel('Imposed budget (units)')
        handles = [Line2D([0], [0], color=STYLES[m][1], marker=STYLES[m][2],
                          linestyle=STYLES[m][3], markerfacecolor='white', label=STYLES[m][0]) for m in METHODS]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .075), ncol=4,
                   frameon=False, columnspacing=2.2)
        caption = ('Shading: full reports are statically infeasible below 26 shared / 48 independent units; '
                   'an unresolved zero-query control spends 0 there.\n'
                   'Points retain failed, unresolved and incomplete requests; lines join evaluated budgets. '
                   'Some method curves coincide.\n'
                   'Point estimates only; paired object-bootstrap contrasts and 95% intervals are in the companion results. '
                   'These are imposed costs, not live or LLM latency.')
        fig.text(.5, .025, caption, ha='center', va='bottom', fontsize=8, color='#444444', linespacing=1.5)
        fig.subplots_adjust(top=.915, bottom=.155, left=.075, right=.985)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {'Title': 'Task completion and imposed acquisition spend',
                    'Creator': 'scripts/plot_partial_benchmark.py',
                    'Description': f'Aggregate input {source_label}; SHA256 {results_sha256}. Absolute all-request levels; no invented confidence intervals.'}
        if output.suffix.lower() == '.svg':
            metadata['Date'] = None
        fig.savefig(output, metadata=metadata, dpi=180)
        plt.close(fig)
    return {'output': output.name, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'profile_cost_budget_cells': 16, 'method_records': 64, 'completion_points': 64, 'mean_spend_points': sum(cell.mean_spend is not None for cell in cells),
            'input_sha256': results_sha256,
            'absolute_intervals_drawn': False}


def load_cells(document):
    """Use only the released clean confirmation aggregate and its actual means."""
    if document.get('status', {}).get('confirmation_executed') is not True:
        raise ValueError('real confirmation results have not been exported; no figure rendered')
    confirmation = document.get('confirmation_clean', {})
    if confirmation.get('grid_validation', {}).get('unique_complete_grid') is not True:
        raise ValueError('confirmation aggregate grid has not passed completeness validation')
    records = confirmation.get('cells')
    if not isinstance(records, list):
        raise ValueError('confirmation_clean.cells must contain the actual aggregate records')
    cells = []
    for row in records:
        if not isinstance(row, dict) or any(row.get(key) != value for key, value in
                (('scenario', 'clean'), ('support', 'logical'), ('missing_semantics', 'unknown'))):
            raise ValueError('plot accepts only clean logical-support unknown-evidence confirmation cells')
        spend = row['mean_imposed_cost_all_requests']
        if spend is not None and (isinstance(spend, bool) or not isinstance(spend, (int, float))):
            raise ValueError('all-request mean spend must be a finite number or unavailable')
        cells.append(Cell(profile=row['profile'], cost=row['cost_scenario'], budget=row['budget'],
                          method=row['method'], n_requested=row['n_requested'], n_objects=row['n_objects'],
                          tasks_complete=row['tasks_complete'], mean_spend=spend))
    by_key = validate_cells(cells)
    n_requested = cells[0].n_requested
    grid = confirmation['grid_validation']
    if grid.get('planned_rows') != 64 * n_requested or grid.get('completed_rows') != 64 * n_requested:
        raise ValueError('confirmation aggregate does not retain every requested cell row')
    contrasts = confirmation.get('paired_contrasts_vs_compiled')
    if not isinstance(contrasts, list):
        raise ValueError('paired companion contrasts are missing')
    expected_contrasts = {key for key in by_key if key[3] != 'compiled'}
    seen = set()
    for row in contrasts:
        key = (row['profile'], row['cost_scenario'], row['budget'], row['method'])
        if key not in expected_contrasts or key in seen or row.get('baseline') != 'compiled':
            raise ValueError('unknown or duplicate paired companion contrast')
        if row.get('n_pairs') != n_requested or row.get('n_objects') != cells[0].n_objects:
            raise ValueError('paired companion contrast does not retain the same requested cohort')
        seen.add(key)
    if seen != expected_contrasts:
        raise ValueError('paired companion contrasts do not cover all prespecified cells')
    return cells


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True, help='released aggregate results JSON')
    parser.add_argument('--schema', type=Path, required=True, help='published schema recorded with plot provenance')
    parser.add_argument('--output', type=Path, required=True, help='standalone .svg or .png output')
    args = parser.parse_args(argv)
    if args.output.suffix.lower() not in ('.svg', '.png'):
        parser.error('--output must use .svg or .png')
    data = args.results.read_bytes()
    schema = args.schema.read_bytes()
    # Scientific grid checks are explicit above; this does not pretend to be a
    # general JSON Schema implementation or add another plotting dependency.
    schema_document = json.loads(schema)
    if not isinstance(schema_document, dict):
        raise ValueError('published results schema must be a JSON object')
    document = json.loads(data)
    version = schema_document.get('properties', {}).get('schema_version', {}).get('const')
    if version is None or document.get('schema_version') != version:
        raise ValueError('aggregate and published schema versions do not match')
    cells = load_cells(document)
    receipt = render_figure(cells, args.output,
        results_sha256=hashlib.sha256(data).hexdigest(), source_label=args.results.name)
    receipt.update({'schema_sha256': hashlib.sha256(schema).hexdigest(),
                    'plot_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'all_request_denominators_retained': True,
                    'paired_intervals_source': 'companion public results; not drawn on absolute means'})
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
