from itertools import combinations
import pytest
from applecider_agents.core.acquisition import CostModel, FiniteWorld, FiniteAcquisition, StateLimitExceeded


def test_shared_backend_charges_work_once():
    costs = CostModel({'classify': 1, 'anomaly': 1, 'context': 1},
                      {'classify': 'flare', 'anomaly': 'flare'}, {'flare': 20})
    assert costs.total(['classify', 'anomaly', 'classify', 'context']) == 23
    assert costs.unique_backends(['classify', 'anomaly']) == ('flare',)


def test_ec2_global_weights_equal_explicit_pair_enumeration():
    worlds = [FiniteWorld({'a': 0, 'b': 0}, 'ordinary', 1),
              FiniteWorld({'a': 0, 'b': 1}, 'rare', 3),
              FiniteWorld({'a': 1, 'b': 0}, 'ordinary', 2),
              FiniteWorld({'a': 1, 'b': 1}, 'rare', 7)]
    p = FiniteAcquisition(('a', 'b'), worlds, CostModel({'a': 1, 'b': 1}), lambda _: False)
    ids = p.compatible({})
    pairs = [(i, j) for i, j in combinations(ids, 2) if worlds[i].decision != worlds[j].decision]
    parent = sum(worlds[i].weight * worlds[j].weight for i, j in pairs)
    assert p.edge_mass(ids) == parent
    for action in p.actions:
        remaining = 0
        for outcome, child in p.partitions(ids, action).items():
            q = sum(worlds[i].weight for i in child) / sum(w.weight for w in worlds)
            edges = sum(worlds[i].weight * worlds[j].weight for i, j in pairs
                        if worlds[i].outcomes[action] == outcome == worlds[j].outcomes[action])
            remaining += q * edges
        assert p.edge_cut(ids, action) == pytest.approx(parent - remaining)
    assert p.ec2_next({}, 2) == 'b'


def test_bounded_dp_keeps_unresolvable_probability_and_minimizes_spend():
    worlds = [FiniteWorld({'cheap': 'valid', 'costly': 'valid'}, 'A', 1),
              FiniteWorld({'cheap': 'failed', 'costly': 'valid'}, 'B', 1),
              FiniteWorld({'cheap': 'failed', 'costly': 'failed'}, 'B', 2)]
    cert = lambda obs: 'valid' in obs.values()
    p = FiniteAcquisition(('cheap', 'costly'), worlds, CostModel({'cheap': 1, 'costly': 2}), cert)
    value = p.exact_dp({}, 3)
    # Costly first resolves1/2; after failure the cheap tool cannot help.
    assert value.action == 'costly'
    assert value.success_probability == .5
    assert value.expected_spent_cost == 2
    assert p.exact_dp({}, 1).success_probability == .25
    assert p.exact_dp({}, 0).action is None


def test_dp_prefers_completion_probability_over_cheap_partial_success():
    worlds = [FiniteWorld({'a': 0, 'b': 1}, 'A', 9), FiniteWorld({'a': 1, 'b': 1}, 'B', 1)]
    p = FiniteAcquisition(('a', 'b'), worlds, CostModel({'a': 1, 'b': 2}),
                          lambda o: o.get('a') == 1 or o.get('b') == 1)
    value = p.exact_dp({}, 2)
    assert value.action == 'b'
    assert value.success_probability == 1
    assert value.expected_spent_cost == 2


def test_prior_support_cannot_supply_certificate():
    p = FiniteAcquisition(('a',), [FiniteWorld({'a': 0}, 'A')], CostModel({'a': 1}),
                          lambda o: o.get('a') == 0)
    assert p.exact_dp({}, 1).action == 'a'
    assert p.ec2_next({}, 1) == 'a'
    assert p.fixed_next({}, 1, ('a',)) == 'a'
    with pytest.raises(ValueError, match='zero mass'):
        p.exact_dp({'a': 1}, 1)


def test_report_certificate_requires_mandatory_fields_after_decision():
    p = FiniteAcquisition(('decision', 'diagnostic'),
                          [FiniteWorld({'decision': 'yes', 'diagnostic': 'actual'}, 'A')],
                          CostModel({'decision': 1, 'diagnostic': 3}),
                          lambda o: set(o) == {'decision', 'diagnostic'},
                          required=lambda o: ('diagnostic',))
    assert p.exact_dp({'decision': 'yes'}, 2).success_probability == 0
    assert p.exact_dp({'decision': 'yes'}, 3).expected_spent_cost == 3
    assert p.ec2_next({'decision': 'yes'}, 3) == 'diagnostic'


def test_no_success_branch_means_stop_without_spending():
    p = FiniteAcquisition(('a',), [FiniteWorld({'a': 'unavailable'}, 'A')],
                          CostModel({'a': 1}), lambda o: False)
    assert p.exact_dp({}, 3).action is None
    assert p.exact_dp({}, 3).expected_spent_cost == 0


def test_state_cap_is_explicit_error():
    p = FiniteAcquisition(('a',), [FiniteWorld({'a': 0}, 'A')], CostModel({'a': 1}),
                          lambda o: bool(o), max_states=1)
    with pytest.raises(StateLimitExceeded):
        p.exact_dp({}, 1)


def test_dense_array_controls_match_object_controls():
    import numpy as np
    from applecider_agents.core.acquisition import ArrayFiniteAcquisition
    worlds = [FiniteWorld({'a': 0, 'b': 0}, 0, 1),
              FiniteWorld({'a': 0, 'b': 1}, 1, 3),
              FiniteWorld({'a': 1, 'b': 0}, 0, 2)]
    costs = CostModel({'a': 1, 'b': 2})
    cert = lambda o: 'b' in o
    obj = FiniteAcquisition(('a', 'b'), worlds, costs, cert)
    arr = ArrayFiniteAcquisition(('a', 'b'), np.array([[0, 0], [0, 1], [1, 0]]),
                                 [0, 1, 0], [1, 3, 2], costs, cert)
    assert arr.edge_cut(arr.compatible({}), 'a') == pytest.approx(obj.edge_cut(obj.compatible({}), 'a'))
    assert arr.exact_dp({}, 3) == obj.exact_dp({}, 3)


def test_independent_scientific_reviewer_ec2_and_dp_oracles():
    # Independently specified by the scientific reviewer, not inferred from code.
    worlds = [FiniteWorld({'query': outcome}, label, weight / 10)
              for outcome, label, weight in ((0, 'A', 1), (1, 'A', 2), (0, 'B', 3), (1, 'B', 4))]
    p = FiniteAcquisition(('query',), worlds, CostModel({'query': 2}), lambda _: False)
    indices = p.compatible({})
    assert p.edge_mass(indices) == pytest.approx(.21)
    children = p.partitions(indices, 'query')
    assert p.edge_mass(children[0]) == pytest.approx(.03)
    assert p.edge_mass(children[1]) == pytest.approx(.08)
    assert p.edge_cut(indices, 'query') == pytest.approx(.15)
    mixed = FiniteAcquisition(('query',),
        [FiniteWorld({'query': 'valid'}, 'A'), FiniteWorld({'query': 'failed'}, 'B')],
        CostModel({'query': 2}), lambda o: o.get('query') == 'valid')
    assert mixed.exact_dp({}, 2).success_probability == .5
    assert mixed.exact_dp({}, 2).expected_spent_cost == 2
    assert mixed.exact_dp({}, 1).success_probability == 0
