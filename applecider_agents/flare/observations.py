"""Opt-in local FLARE observation acquisition with fixed prefix/snapshot ownership."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import time
import uuid
from typing import Literal

import numpy as np
import pandas as pd
from ..core.evidence import EvidenceScope, ToolObservation
from ..core.observations import RecordedSession
from ..core import config as C, data as DATA, object_tools as O
from . import tools as T
from flare import data as D


def _finite_json(value):
    if isinstance(value, dict):
        return {str(k): _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def _events(array, horizon_days):
    arr = np.array(array, copy=True)
    if arr.ndim != 2 or arr.shape[1] not in (14, 15):
        raise ValueError("events must have14 science columns, optionally a15th ignored label column")
    # Validate time coordinates before selecting the prefix. No future flux,
    # uncertainty, band or color channel participates in early validation.
    times = np.asarray(arr[:, D.COL["dt"]], dtype=float)
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("relative time coordinates must be finite and nonnegative")
    # Discard the ignored truth channel before numeric conversion/validation.
    science = np.array(arr[times <= horizon_days, :14], dtype=float, copy=True)
    arr = np.column_stack((science, np.full(len(science), -1.0)))
    if not np.isfinite(arr[:, :8]).all() or (arr[:, D.COL["dt"]] < 0).any():
        raise ValueError("event time/band/flux channels must be finite, with nonnegative relative time")
    if (arr[:, D.COL["logflux_err"]] <= 0).any() or not np.isin(arr[:, D.COL["band_id"]], (0, 1, 2)).all():
        raise ValueError("invalid event uncertainty or band")
    return arr


def _local_context(obj_id, data_dir):
    """Only documented scientific table columns; never spectroscopy/catalogue labels."""
    ctx = {"_source": "local_benchmark"}
    files, matched_files = [], []
    for name, columns in DATA.TABLES.items():
        path = data_dir / name if data_dir is not None else None
        if path is None or not path.exists():
            continue
        table = pd.read_csv(path, usecols=lambda c: c == "obj_id" or c in columns)
        row = table.loc[table.obj_id == obj_id]
        files.append(path.name)
        if len(row):
            matched_files.append(path.name)
            for column in columns:
                if column in row:
                    value = row[column].iloc[0]
                    if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                        ctx[column] = float(value)
    return ctx, files, matched_files


def collect_flare(obj_id: str, horizon_days: float = 100.0, *, session_id: str | None = None,
                  events_array=None, context: dict | None = None,
                  event_dirs=None, data_dir: str | Path | None = None) -> RecordedSession:
    """Acquire real tools locally, without catalogue truth, live history or posting.

    Static context dates and absolute event origin are unknown. Full history is
    unavailable, never a fabricated negative. Inputs may be supplied directly for
    an application-owned snapshot. All science caches respect its new revision.
    This callable uses blind policy configuration; it does not call an LLM.
    """
    scope = EvidenceScope(obj_id=obj_id, horizon_days=horizon_days, session_id=session_id or uuid.uuid4().hex)
    observations = []
    started = time.perf_counter()
    costs = {"kind": "actual_local_acquisition", "logical_calls": 0,
             "feature_calls": 0, "inference_calls": 0, "backend_records": []}
    metadata = {"blind": True, "catalogue": "policy_disabled", "history": "unavailable_offline",
                "absolute_epoch_known": False, "static_catalogue_availability_known": False,
                "relative_horizon_days": scope.horizon_days}
    scientific_input = {}
    try:
        source = "supplied"
        if events_array is None:
            dirs = [Path(p) for p in event_dirs] if event_dirs is not None else C.EVENT_DIRS
            path = next((p / f"{obj_id}.npz" for p in dirs if (p / f"{obj_id}.npz").exists()), None)
            if path is None:
                raise FileNotFoundError(f"no local events for {obj_id}")
            with np.load(path, allow_pickle=False) as archive:
                events_array = archive["data"]
            source = path.name
        events = _events(events_array, scope.horizon_days)
        prefix = events[events[:, D.COL["dt"]] <= scope.horizon_days]
        if context is None:
            ctx, files, matched_files = _local_context(obj_id, Path(data_dir) if data_dir is not None else C.DATA_DIR)
            context_available = any(key != "_source" for key in ctx)
            metadata["context_files"] = files
            metadata["context_matched_files"] = matched_files
        else:
            # Only documented science columns enter features or exported context.
            allowed = {c for cols in DATA.TABLES.values() for c in cols}
            ctx = {k: float(v) for k, v in context.items() if k in allowed
                   and isinstance(v, (int, float, np.number)) and np.isfinite(v)}
            ctx["_source"] = "provided"
            context_available = bool(allowed.intersection(ctx))
            metadata["context_files"] = []
        allowed_context = {c for cols in DATA.TABLES.values() for c in cols}
        metadata["raw_context_missing_fields"] = sorted(allowed_context - ctx.keys())
        metadata["context_flag_semantics"] = "successful upstream boolean flags preserved; missing raw values remain explicit metadata"
        scientific_input = {"events": _finite_json(prefix[:, :14]), "event_columns": list(D.COL)[:14],
                            "context": _finite_json(ctx), "event_source": source,
                            "color_channels": "archival input; actual feature extraction recomputes prefix colors"}
    except Exception as exc:
        metadata["input_error"] = f"{type(exc).__name__}: {exc}"
        for name in ("flare_classify", "flare_anomaly", "get_photometry", "get_context"):
            observations.append(ToolObservation(tool=name, scope=scope,
                inputs={"obj_id": obj_id, "horizon_days": scope.horizon_days}, status="error",
                payload=None, provenance={"source": "local_input_validation"},
                availability={"reason": metadata["input_error"]}))
    else:
        def observe_work(record):
            costs["feature_calls" if record["backend"] == "feature_frame" else "inference_calls"] += 1
            costs["backend_records"].append(record)
        with DATA.snapshot(obj_id, events, ctx), T.scoped_cache(observe_work):
            for name, tool in (("flare_classify", T.flare_classify), ("flare_anomaly", T.flare_anomaly),
                               ("get_photometry", O.get_photometry), ("get_context", O.get_context)):
                if name == "get_context" and not context_available:
                    observations.append(ToolObservation(tool=name, scope=scope,
                        inputs={"obj_id": obj_id, "horizon_days": scope.horizon_days},
                        status="unavailable", payload=None,
                        provenance={"source": "local_context_availability"},
                        availability={"reason": "no_supplied_or_local_context", "no_search_performed": True}))
                    continue
                tick = time.perf_counter()
                before = {key: costs[key] for key in ("feature_calls", "inference_calls")}
                cache_hit = T.cached_result(obj_id, scope.horizon_days)
                inputs = {"obj_id": obj_id, "horizon_days": scope.horizon_days}
                try:
                    payload = _finite_json(tool.call(inputs))
                    status = "error" if "error" in payload else "success"
                except Exception as exc:
                    payload, status = None, "error"
                    availability = {"reason": f"{type(exc).__name__}: {exc}"}
                else:
                    availability = {}
                costs["logical_calls"] += 1
                observations.append(ToolObservation(tool=name, scope=scope, inputs=inputs, status=status,
                    payload=payload, provenance={"source": "actual_local_upstream_tool", "event_source": source,
                    "relative_photometric_horizon_days": scope.horizon_days,
                    "absolute_epoch_known": False, "static_catalogue_availability_known": False,
                    "wall_seconds": time.perf_counter() - tick, "shared_flare_cache_hit": cache_hit,
                    "backend_work_delta": {key: costs[key] - before[key] for key in before}}, availability=availability))
    observations.append(ToolObservation(tool="light_curve_history", scope=scope, inputs={"obj_id": obj_id},
        status="unavailable", payload=None, provenance={"source": "offline_availability_contract"},
        availability={"reason": "no_aligned_timestamped_history", "no_search_performed": True}))
    costs["wall_seconds"] = time.perf_counter() - started
    return RecordedSession(scope=scope, observations=tuple(observations), scientific_input=scientific_input,
                           metadata=metadata, costs=_finite_json(costs))


def flare_partial_triage(obj_id: str, horizon_days: float = 100.0, profile: Literal["decision_only", "legacy_report"] = "decision_only") -> dict:
    """Opt-in offline blind triage with possible decisions and explicit unknown evidence.

    Uses local event/context inputs only; full history is unavailable. Returns a
    partial certificate separate from the legacy publishable Report. profile is
    decision_only or legacy_report. Relative prefix scope is not historical as-of.
    """
    from .partial import evaluate_flare
    session = collect_flare(obj_id, horizon_days)
    result = evaluate_flare(session.scope, session.observations, profile=profile, blind=True)
    return {"result": result.model_dump(mode="json"), "scope": session.scope.model_dump(mode="json"),
            "metadata": session.metadata, "costs": session.costs}
