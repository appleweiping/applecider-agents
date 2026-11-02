"""Finite logical support for partial observations, independent of any prior.

The core accepts an explicit finite state table. Domain-specific payload mapping
and report templates stay with the model adapter. No probability or cost changes
which rows remain logically admissible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, NamedTuple, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Decision(NamedTuple):
    verdict: str
    priority: int


@dataclass(frozen=True)
class FinitePolicy:
    policy_id: str
    predicates: tuple[str, ...]
    states: np.ndarray
    decisions: Sequence[Decision]
    tool_predicates: Mapping[str, tuple[str, ...]]
    domain_kind: str = "explicit_finite_domain"
    domain_notes: str = "Exact only relative to the explicitly supplied state domain."
    _decision_codes: np.ndarray = field(init=False, repr=False, compare=False)
    _unique_decisions: tuple[Decision, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        states = np.asarray(self.states)
        if states.dtype != np.bool_ or states.ndim != 2 or states.shape[1] != len(self.predicates):
            raise ValueError("states must be a Boolean matrix with one column per predicate")
        if len(set(self.predicates)) != len(self.predicates) or len(self.decisions) != len(states):
            raise ValueError("predicate names must be unique and decisions must align with rows")
        if any(p not in self.predicates for names in self.tool_predicates.values() for p in names):
            raise ValueError("tool groups refer to an undeclared predicate")
        decisions = tuple(Decision(*d) for d in self.decisions)
        if any(not isinstance(d.verdict, str) or type(d.priority) is not int or not 0 <= d.priority <= 3 for d in decisions):
            raise ValueError("each decision requires a verdict and integer priority 0..3")
        states = states.copy()
        states.flags.writeable = False
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "tool_predicates", dict(self.tool_predicates))
        unique = tuple(sorted(set(decisions)))
        decision_index = {d: i for i, d in enumerate(unique)}
        codes = np.array([decision_index[d] for d in decisions], dtype=np.int32)
        keys = [int.from_bytes(row.tobytes(), "little") for row in np.packbits(states, axis=1, bitorder="little")]
        lookup = dict(zip(keys, range(len(keys))))
        if any(decisions[i] != decisions[lookup[key]] for i, key in enumerate(keys)):
            raise ValueError("one complete state cannot have conflicting policy decisions")
        codes.flags.writeable = False
        object.__setattr__(self, "_decision_codes", codes)
        object.__setattr__(self, "_unique_decisions", unique)

    @classmethod
    def from_worlds(cls, policy_id: str, worlds: Iterable[Mapping[str, bool]],
                    decision: Callable[[Mapping[str, bool]], Decision], *,
                    predicates: tuple[str, ...], tool_predicates: Mapping[str, tuple[str, ...]],
                    domain_kind: str = "explicit_finite_domain", domain_notes: str = "Explicit finite states."):
        worlds = tuple(worlds)
        if any(set(w) != set(predicates) or any(type(v) is not bool for v in w.values()) for w in worlds):
            raise ValueError("each world must supply every declared Boolean predicate exactly once")
        states = np.array([[w[p] for p in predicates] for w in worlds], dtype=bool).reshape(len(worlds), len(predicates))
        return cls(policy_id, predicates, states, tuple(decision(w) for w in worlds),
                   tool_predicates, domain_kind, domain_notes)

    def worlds(self):
        for row in self.states:
            yield dict(zip(self.predicates, map(bool, row)))


class Reachability(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: Literal["decided", "unresolved", "inconsistent"]
    possible_decisions: tuple[Decision, ...]
    missing_predicates: tuple[str, ...]
    admissible_count: int
    domain_kind: str


class PartialResult(BaseModel):
    """A diagnostic result, deliberately separate from the publishable Report."""
    model_config = ConfigDict(frozen=True)
    scope: Any
    policy_id: str
    profile: Literal["decision_only", "legacy_report"]
    status: Literal["decided", "unresolved", "inconsistent"]
    possible_decisions: tuple[Decision, ...]
    decision_certified: bool
    report_complete: bool
    task_complete: bool
    missing_evidence: tuple[str, ...]
    missing_report_fields: tuple[str, ...]
    used_evidence: dict[str, list[str]] = Field(default_factory=dict)
    rejected_evidence: dict[str, list[str]] = Field(default_factory=dict)
    report: dict[str, Any] | None = None
    domain_kind: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def decision(self) -> Decision | None:
        return self.possible_decisions[0] if self.decision_certified else None


def evaluate_partial(policy: FinitePolicy, constraints: Mapping[str, bool], *,
                     conflicts: Sequence[str] = (), include_missing: bool = True) -> Reachability:
    unknown = set(constraints) - set(policy.predicates)
    if unknown or any(type(v) is not bool for v in constraints.values()):
        raise ValueError("constraints must use declared predicate names and Boolean values")
    mask = np.ones(len(policy.states), dtype=bool)
    if conflicts:
        mask[:] = False
    for i, name in enumerate(policy.predicates):
        if name in constraints:
            mask &= policy.states[:, i] == constraints[name]
    rows = np.flatnonzero(mask)
    possible = tuple(policy._unique_decisions[int(i)] for i in np.unique(policy._decision_codes[mask]))
    status = "inconsistent" if not len(rows) else "decided" if len(possible) == 1 else "unresolved"
    # Conservative query dependencies, not a minimal explanation: interactions
    # can matter even when neither one-step outcome set shrinks (e.g. XOR).
    missing = [p for i, p in enumerate(policy.predicates) if p not in constraints and
               np.any(policy.states[rows, i] != policy.states[rows[0], i])] if include_missing and len(possible) > 1 else []
    return Reachability(status=status, possible_decisions=possible, missing_predicates=tuple(missing),
                        admissible_count=len(rows), domain_kind=policy.domain_kind)
