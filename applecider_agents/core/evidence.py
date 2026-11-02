"""Captured evidence for an opt-in partial-policy evaluation.

A session identifies one effective input snapshot, not an astronomical epoch.
Callers replace it after reseeding. Captures own copies of supplied containers;
the frozen records should be treated as read-only after construction.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvidenceScope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    obj_id: str = Field(min_length=1)
    horizon_days: float = Field(ge=0, allow_inf_nan=False)
    session_id: str = Field(min_length=1)


class ToolObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    tool: str = Field(min_length=1)
    scope: EvidenceScope
    inputs: dict[str, Any]
    status: Literal["success", "error", "unavailable", "not_required"]
    payload: dict[str, Any] | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    availability: dict[str, Any] = Field(default_factory=dict)

    @field_validator("inputs", "payload", "provenance", "availability", mode="before")
    @classmethod
    def capture(cls, value):
        return deepcopy(value)

    def unusable_reason(self, scope: EvidenceScope) -> str | None:
        """A transport/scope failure is unknown, never a negative measurement."""
        if self.scope != scope:
            return "object, horizon, or session does not match the request"
        if self.status != "success":
            return f"tool status is {self.status}"
        if self.payload is None or "error" in self.payload:
            return "tool did not return a successful scientific payload"
        if self.inputs.get("obj_id") != scope.obj_id:
            return "actual input object is missing or does not match the request"
        for name, values in (("input", self.inputs), ("payload", self.payload)):
            if "obj_id" in values and values["obj_id"] != scope.obj_id:
                return f"{name} object does not match the request"
            if "horizon_days" in values:
                value = values["horizon_days"]
                if not finite_number(value) or float(value) != scope.horizon_days:
                    return f"{name} horizon does not match the request"
            if "session_id" in values and values["session_id"] != scope.session_id:
                return f"{name} session does not match the request"
        return None


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def finite_science(value: Any) -> bool:
    """Null is missing, but an actual non-finite measurement is invalid."""
    if isinstance(value, float):
        return isfinite(value)
    if isinstance(value, dict):
        return all(finite_science(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_science(v) for v in value)
    return True
