"""Captured scientific observations, separate from publishable legacy reports."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .evidence import EvidenceScope, ToolObservation


class RecordedSession(BaseModel):
    """One fixed object/prefix/snapshot, with acquisition costs distinct from replay."""
    model_config = ConfigDict(frozen=True)
    scope: EvidenceScope
    observations: tuple[ToolObservation, ...]
    scientific_input: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    costs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("scientific_input", "metadata", "costs", mode="before")
    @classmethod
    def capture(cls, value):
        return copy.deepcopy(value)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.model_dump(mode="json"), indent=2, allow_nan=False) + "\n")


def load_raw_snapshot(path: str | Path, *, context_variant: str = "prefix") -> RecordedSession:
    """Read genuine raw payloads without invoking any science or network backend.

    The legacy100 variant deliberately retains its effective100d input scope and
    must not certify an early prefix. Replay is never a measurement of live cost.
    """
    if context_variant not in ("prefix", "legacy100"):
        raise ValueError("context_variant must be prefix or legacy100")
    path = Path(path)
    raw = json.loads(path.read_text())
    scope = EvidenceScope(obj_id=raw["obj_id"], horizon_days=raw["horizon_days"],
                          session_id=f"raw:{raw['obj_id']}:h{raw['horizon_days']:g}:{context_variant}")
    selected = f"get_context_{context_variant}"
    observations = []
    for record in raw["observations"]:
        tool = record["tool"]
        if tool.startswith("get_context_"):
            if tool != selected:
                continue
            tool = "get_context"
        effective = record.get("effective_photometric_horizon_days", scope.horizon_days)
        observation_scope = scope if effective == scope.horizon_days else EvidenceScope(obj_id=scope.obj_id, horizon_days=effective, session_id=scope.session_id)
        status = record.get("status", "error")
        if status not in ("success", "error", "unavailable", "not_required"):
            status = "error"
        inputs = copy.deepcopy(record.get("input", {}))
        if tool == "get_context":
            inputs["horizon_days"] = effective
        observations.append(ToolObservation(
            tool=tool, scope=observation_scope, inputs=inputs, status=status,
            payload=record.get("result"), provenance={
                "source": record.get("source"), "replay_file": path.name,
                "relative_photometric_horizon_days": effective,
                "absolute_epoch_known": False, "static_catalogue_availability_known": False,
                "recorded_acquisition": {k: record.get(k) for k in
                    ("wall_seconds", "backend_work_delta", "backend_records", "scientific_input_ref", "event_file_sha256")}},
            availability={"reason": record.get("reason"), "no_search_performed": status == "unavailable"}))
    scientific_input = copy.deepcopy(raw.get("scientific_input", {}))
    if "event_source_path" in scientific_input:
        scientific_input["event_source"] = Path(scientific_input.pop("event_source_path")).name
    return RecordedSession(scope=scope, observations=tuple(observations),
        scientific_input=scientific_input,
        metadata={"source": "raw_snapshot_replay", "context_variant": context_variant, "blind": True,
                  "catalogue": "policy_disabled", "history": "unavailable_offline",
                  "absolute_epoch_known": False, "static_catalogue_availability_known": False},
        costs={"kind": "recorded_acquisition_not_live_replay", "recorded_session_wall_seconds": raw.get("wall_seconds")})
