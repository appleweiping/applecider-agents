"""Plot input guards only: fixture numbers are never rendered as a result figure."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def plot_module():
    path = Path(__file__).resolve().parents[1] / 'scripts/plot_partial_benchmark.py'
    spec = importlib.util.spec_from_file_location('plot_partial_benchmark_test', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixture_cells(module):
    return [module.Cell(profile, cost, budget, method, 4, 1, 0, 0.0)
            for profile, cost in module.ROWS for budget in module.BUDGETS for method in module.METHODS]


def test_plot_requires_complete_unique_all_method_grid(plot_module):
    cells = fixture_cells(plot_module)
    assert len(plot_module.validate_cells(cells)) == 64
    with pytest.raises(ValueError, match='exactly 16'):
        plot_module.validate_cells(cells[:-1])
    with pytest.raises(ValueError, match='duplicate'):
        plot_module.validate_cells(cells + [cells[0]])


def test_plot_rejects_denominator_changes_and_false_below_floor_completion(plot_module):
    cells = fixture_cells(plot_module)
    with pytest.raises(ValueError, match='cohort'):
        plot_module.validate_cells([replace(cells[0], n_requested=8, n_objects=2)] + cells[1:])
    with pytest.raises(ValueError, match='floor'):
        plot_module.validate_cells([replace(cells[0], tasks_complete=1)] + cells[1:])


def test_unavailable_cost_remains_missing_not_zero(plot_module):
    cells = fixture_cells(plot_module)
    cells[0] = replace(cells[0], mean_spend=None)
    record = plot_module.validate_cells(cells)[(cells[0].profile, cells[0].cost, cells[0].budget, cells[0].method)]
    assert record.mean_spend is None


def test_plot_refuses_unexecuted_confirmation_or_unvalidated_grid(plot_module):
    with pytest.raises(ValueError, match='not been exported'):
        plot_module.load_cells({'status': {'confirmation_executed': False}})
    with pytest.raises(ValueError, match='completeness'):
        plot_module.load_cells({'status': {'confirmation_executed': True}, 'confirmation_clean': {}})


def aggregate_fixture(module):
    # Small software fixture only. These records are never passed to rendering.
    cells = fixture_cells(module)
    records = [dict(profile=c.profile, cost_scenario=c.cost, budget=c.budget, method=c.method,
                    scenario='clean', support='logical', missing_semantics='unknown',
                    n_requested=c.n_requested, n_objects=c.n_objects, tasks_complete=c.tasks_complete,
                    mean_imposed_cost_all_requests=c.mean_spend) for c in cells]
    contrasts = [dict(profile=c.profile, cost_scenario=c.cost, budget=c.budget, method=c.method,
                      baseline='compiled', n_pairs=c.n_requested, n_objects=c.n_objects)
                 for c in cells if c.method != 'compiled']
    return {'status': {'confirmation_executed': True}, 'confirmation_clean': {
        'grid_validation': {'unique_complete_grid': True, 'planned_rows': 256, 'completed_rows': 256},
        'cells': records, 'paired_contrasts_vs_compiled': contrasts}}


def test_plot_adapter_preserves_absolute_metrics_and_rejects_pair_or_grid_mismatch(plot_module):
    import copy
    document = aggregate_fixture(plot_module)
    assert plot_module.load_cells(document) == fixture_cells(plot_module)
    duplicate = copy.deepcopy(document)
    duplicate['confirmation_clean']['paired_contrasts_vs_compiled'].append(
        duplicate['confirmation_clean']['paired_contrasts_vs_compiled'][0])
    with pytest.raises(ValueError, match='duplicate paired'):
        plot_module.load_cells(duplicate)
    mismatched = copy.deepcopy(document)
    mismatched['confirmation_clean']['grid_validation']['completed_rows'] -= 1
    with pytest.raises(ValueError, match='every requested'):
        plot_module.load_cells(mismatched)
