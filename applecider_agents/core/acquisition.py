"""Finite acquisition comparators, with logical certification separate from priors.

EC2 and stochastic function evaluation are established methods. This module
implements controls, not a novel acquisition algorithm. Wall time is not cost.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from math import isfinite
from time import perf_counter
from typing import Callable, Hashable, Mapping, Sequence

Outcome = Hashable
Observed = Mapping[str, Outcome]
Certificate = Callable[[Observed], bool]


@dataclass(frozen=True)
class FiniteWorld:
    """A latent decision and complete observable outcome vector in a finite prior."""
    outcomes: Mapping[str, Outcome]
    decision: Hashable
    weight: float = 1.0


@dataclass(frozen=True)
class CostModel:
    """Charge logical wrappers and each shared backend separately, in integer units."""
    call_costs: Mapping[str, int]
    backend_for: Mapping[str, str] = field(default_factory=dict)
    backend_costs: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for value in (*self.call_costs.values(), *self.backend_costs.values()):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('costs must be nonnegative integer units')

    def increment(self, action: str, attempted: Sequence[str]) -> int:
        if action in attempted:
            return 0
        group = self.backend_for.get(action)
        used = {self.backend_for.get(a) for a in attempted}
        return self.call_costs[action] + (
            self.backend_costs.get(group, 0) if group is not None and group not in used else 0
        )

    def total(self, attempted: Sequence[str]) -> int:
        seen: list[str] = []
        total = 0
        for action in attempted:
            total += self.increment(action, seen)
            seen.append(action)
        return total

    def unique_backends(self, attempted: Sequence[str]) -> tuple[str, ...]:
        return tuple(sorted({self.backend_for[a] for a in attempted if a in self.backend_for}))


@dataclass(frozen=True)
class PlanValue:
    success_probability: float
    expected_spent_cost: float
    action: str | None


class StateLimitExceeded(RuntimeError):
    pass


class FiniteAcquisition:
    """Planners over a declared finite prior and an independent certificate oracle.

    ``certificate`` MUST check the full logical domain and report obligations;
    it must never infer admissibility from the positive-weight prior support.
    Zero weights are permitted for the explicitly labeled empirical-support
    ablation. They do not remove logical worlds from the certificate callback.
    An outcome may encode failure, but that failure then remains in the joint
    prior. The callback decides which outcomes are usable scientific evidence.
    """
    def __init__(self, actions: Sequence[str], worlds: Sequence[FiniteWorld],
                 costs: CostModel, certificate: Certificate,
                 required: Callable[[Observed], Sequence[str]] | None = None,
                 *, max_states: int = 200_000, max_seconds: float = 30.0):
        self.actions = tuple(actions)
        if len(set(self.actions)) != len(self.actions) or not self.actions:
            raise ValueError('actions must be nonempty and unique')
        if set(self.actions) != set(costs.call_costs):
            raise ValueError('cost model must cover exactly the actions')
        self.worlds = tuple(worlds)
        if not self.worlds:
            raise ValueError('finite prior must be nonempty')
        for world in self.worlds:
            if set(world.outcomes) != set(self.actions):
                raise ValueError('each prior world must specify every action outcome')
            if not isfinite(world.weight) or world.weight < 0:
                raise ValueError('prior weights must be finite and nonnegative')
            for outcome in world.outcomes.values():
                hash(outcome)
        if sum(w.weight for w in worlds) <= 0:
            raise ValueError('finite prior must have positive total weight')
        self.costs, self.certificate = costs, certificate
        self.required = required or (lambda _: ())
        self.max_states = max_states
        self.dp_states = 0
        self.max_seconds = max_seconds
        self._dp_cache = {}

    def compatible(self, observed: Observed) -> tuple[int, ...]:
        if set(observed) - set(self.actions):
            raise ValueError('unknown observed action')
        return tuple(i for i, world in enumerate(self.worlds)
                     if world.weight > 0 and all(world.outcomes[a] == y for a, y in observed.items()))

    def partitions(self, indices: Sequence[int], action: str) -> dict[Outcome, tuple[int, ...]]:
        groups: dict[Outcome, list[int]] = defaultdict(list)
        for i in indices:
            groups[self.worlds[i].outcomes[action]].append(i)
        return {outcome: tuple(ids) for outcome, ids in groups.items()}

    def mass(self, indices: Sequence[int]) -> float:
        return sum(self.worlds[i].weight for i in indices)

    def edge_mass(self, indices: Sequence[int]) -> float:
        """Cross-decision edge mass, with the same GLOBAL weights in all subsets."""
        masses: dict[Hashable, float] = defaultdict(float)
        for i in indices:
            masses[self.worlds[i].decision] += self.worlds[i].weight
        values = list(masses.values())
        return sum(x * y for j, x in enumerate(values) for y in values[j + 1:])

    def edge_cut(self, indices: Sequence[int], action: str) -> float:
        # Every observed outcome removes all edges outside its version space.
        # q_y is normalized separately; child edge masses are NOT renormalized.
        parent_mass = self.mass(indices)
        if parent_mass <= 0:
            return 0.0
        expected_remaining = sum(self.mass(child) / parent_mass * self.edge_mass(child)
                                 for child in self.partitions(indices, action).values())
        return self.edge_mass(indices) - expected_remaining

    def fixed_next(self, observed: Observed, budget: int, order: Sequence[str]) -> str | None:
        """Compiled short circuit: same logical certificate after every acquisition."""
        if tuple(sorted(order)) != tuple(sorted(self.actions)):
            raise ValueError('fixed order must be a permutation of actions')
        if self.certificate(observed):
            return None
        return next((a for a in order if a not in observed and
                     self.costs.increment(a, tuple(observed)) <= budget), None)

    def ec2_next(self, observed: Observed, budget: int) -> str | None:
        if self.certificate(observed):
            return None
        indices = self.compatible(observed)
        affordable = [a for a in self.actions if a not in observed and
                      self.costs.increment(a, tuple(observed)) <= budget]
        if not affordable:
            return None
        required = set(self.required(observed))
        scores = {}
        for action in affordable:
            cost = self.costs.increment(action, tuple(observed))
            cut = self.edge_cut(indices, action)
            scores[action] = float('inf') if cost == 0 and cut > 0 else cut / max(cost, 1)
        if max(scores.values()) > 0:
            return max(affordable, key=lambda a: (scores[a], -self.actions.index(a)))
        # A one-class prior is not a logical certificate. Mandatory fields still
        # cost something; zero-support fallback acquires rather than certifies.
        return min(affordable, key=lambda a: (a not in required,
                   self.costs.increment(a, tuple(observed)), self.actions.index(a)))

    def exact_dp(self, observed: Observed, budget: int) -> PlanValue:
        """Bounded DP maximizing valid task completion, then minimizing spend.

        Exactness is only with respect to this finite joint prior, integer cost
        model, supplied certificate and budget. Impossible-to-complete branches
        contribute success=0, not a dropped denominator or finite conditional
        cost-to-certificate. Unresolved stopping is always permitted.
        """
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError('budget must be a nonnegative integer')
        start = tuple(sorted(observed.items(), key=lambda item: self.actions.index(item[0])))
        self.dp_states = 0
        started = perf_counter()

        @lru_cache(maxsize=None)
        def solve(items: tuple[tuple[str, Outcome], ...], remaining: int) -> PlanValue:
            key = (items, remaining)
            if key in self._dp_cache:
                return self._dp_cache[key]
            self.dp_states += 1
            if perf_counter() - started > self.max_seconds:
                raise StateLimitExceeded(f'finite DP exceeded {self.max_seconds} wall seconds')
            if self.dp_states > self.max_states:
                raise StateLimitExceeded(f'finite DP exceeded {self.max_states} states')
            current = dict(items)
            if self.certificate(current):
                result = PlanValue(1.0, 0.0, None)
                self._dp_cache[key] = result
                return result
            indices = self.compatible(current)
            total_mass = self.mass(indices)
            if total_mass <= 0:
                raise ValueError('observed outcomes have zero mass under the declared finite prior')
            best = PlanValue(0.0, 0.0, None)
            for action in self.actions:
                if action in current:
                    continue
                cost = self.costs.increment(action, tuple(current))
                if cost > remaining:
                    continue
                success, spent = 0.0, float(cost)
                for outcome, child in self.partitions(indices, action).items():
                    child_items = tuple(sorted((*items, (action, outcome)),
                                              key=lambda item: self.actions.index(item[0])))
                    value = solve(child_items, remaining - cost)
                    probability = self.mass(child) / total_mass
                    success += probability * value.success_probability
                    spent += probability * value.expected_spent_cost
                candidate = PlanValue(success, spent, action)
                if (candidate.success_probability > best.success_probability + 1e-12 or
                    (abs(candidate.success_probability - best.success_probability) <= 1e-12 and
                     candidate.expected_spent_cost < best.expected_spent_cost - 1e-12)):
                    best = candidate
            self._dp_cache[key] = best
            return best

        return solve(start, budget)


class ArrayFiniteAcquisition(FiniteAcquisition):
    """The same finite controls using dense arrays for larger logical supports.

    Outcome codes are integers with caller-defined meanings (including explicit
    failure codes). This changes storage and counting, not the DP/EC2 objectives.
    """
    def __init__(self, actions, outcome_table, decisions, weights, costs,
                 certificate, required=None, *, max_states=200_000, max_seconds=30.0):
        import numpy as np
        self.actions = tuple(actions)
        self.outcome_table = np.asarray(outcome_table, dtype=np.int64)
        self.decision_codes = np.asarray(decisions, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=float)
        n = len(self.weights)
        if (self.outcome_table.shape != (n, len(self.actions)) or
            self.decision_codes.shape != (n,) or n == 0):
            raise ValueError('finite table dimensions must align')
        if (len(set(self.actions)) != len(self.actions) or not self.actions or
            set(self.actions) != set(costs.call_costs)):
            raise ValueError('unique actions and exact cost coverage required')
        if not np.all(np.isfinite(self.weights)) or np.any(self.weights < 0) or self.weights.sum() <= 0:
            raise ValueError('finite nonnegative weights with positive mass required')
        self.costs, self.certificate = costs, certificate
        self.required = required or (lambda _: ())
        self.max_states, self.dp_states = max_states, 0
        self.max_seconds = max_seconds
        self._dp_cache = {}
        self._outcome_masks = [{int(y): self.outcome_table[:, j] == y
                                for y in np.unique(self.outcome_table[:, j])}
                               for j in range(len(self.actions))]
        self._compatible_cache = {}
        self._edge_cache = {}

    def compatible(self, observed):
        import numpy as np
        key = tuple(sorted(observed.items()))
        if key not in self._compatible_cache:
            mask = self.weights > 0
            for action, outcome in key:
                mask = mask & self._outcome_masks[self.actions.index(action)].get(
                    outcome, np.zeros(len(self.weights), dtype=bool))
            self._compatible_cache[key] = np.flatnonzero(mask)
        return self._compatible_cache[key]

    def partitions(self, indices, action):
        import numpy as np
        values = self.outcome_table[indices, self.actions.index(action)]
        return {int(y): indices[values == y] for y in np.unique(values)}

    def mass(self, indices):
        return float(self.weights[indices].sum())

    def edge_mass(self, indices):
        import numpy as np
        masses = np.bincount(self.decision_codes[indices], weights=self.weights[indices])
        # Sum pairwise class mass products without subtracting nearly equal sums.
        return float(sum(x * masses[j + 1:].sum() for j, x in enumerate(masses)))
