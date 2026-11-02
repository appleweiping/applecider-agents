"""Object access: event arrays, coordinates, catalogue context, photometric history.

Benchmark tables (catalogue and the four context blocks) come from the public flare
repository and are cached under APPLECIDER_AGENTS_HOME; a local APPLECIDER_DATA_DIR
overrides them. Light curves come from local event files when APPLECIDER_EVENTS_DIRS is
set, otherwise from the broker archive through `flare.fetch` (BOOM when configured).
Live catalogue queries (`flare.context`) fill the context for objects outside the benchmark.
Everything returned is plain Python so it can go into a tool result.
"""
from __future__ import annotations
import urllib.request
import copy
from contextlib import contextmanager
from contextvars import ContextVar
import itertools
from functools import lru_cache
from pathlib import Path
import numpy as np, pandas as pd
from . import config as C
from flare import data as D                                   # AppleCiDEr 15-column event format
from flare.config import HOST_COLS, GW_COLS, POS_COLS

TABLES = {"hosts.csv": HOST_COLS, "hosts_photoz.csv": ["z_phot", "z_spec"], "context_gaia_wise.csv": GW_COLS, "ps1_position.csv": POS_COLS}
_SEEDED: dict = {}
_REVISIONS: dict[str, int] = {}
_SNAPSHOT = ContextVar("applecider_snapshot", default=None)
_SNAPSHOT_IDS = itertools.count()


def seed(obj_id: str, events_array, context: dict) -> None:
    """Photometry and context supplied by the caller (e.g. the FLARE SkyPortal service, which
    already holds them) to be used for obj_id instead of a fetch."""
    arr = np.array(events_array, copy=True)
    arr.setflags(write=False)
    _SEEDED[obj_id] = (arr, copy.deepcopy(context))
    _REVISIONS[obj_id] = _REVISIONS.get(obj_id, 0) + 1


def revision(obj_id: str) -> int | tuple[str, int]:
    """Monotone identity of supplied observations, for process-local science caches."""
    snapshot = _SNAPSHOT.get()
    if snapshot is not None and snapshot[0] == obj_id:
        return snapshot[3]
    return _REVISIONS.get(obj_id, 0)


def unseed(obj_id: str) -> None:
    """Stop overriding an object; invalidate results from the previous supplied snapshot."""
    _SEEDED.pop(obj_id, None)
    _REVISIONS[obj_id] = _REVISIONS.get(obj_id, 0) + 1


@contextmanager
def snapshot(obj_id: str, events_array, context: dict):
    """Context-local immutable supplied data; legacy/global readers are unaffected."""
    arr = np.array(events_array, copy=True)
    arr.setflags(write=False)
    token = _SNAPSHOT.set((obj_id, arr, copy.deepcopy(context), ("snapshot", next(_SNAPSHOT_IDS))))
    try:
        yield
    finally:
        _SNAPSHOT.reset(token)


def benchmark_file(name: str) -> Path | None:
    """A benchmark table: local data dir if set, else the pinned public copy, cached once."""
    if C.DATA_DIR and (C.DATA_DIR / name).exists():
        return C.DATA_DIR / name
    target = C.CACHE / "benchmark" / name
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(f"{C.FLARE_BENCHMARK_URL}/{name}", target)
        return target
    except Exception:
        return None


def local_event_path(obj_id: str) -> Path | None:
    for d in C.EVENT_DIRS:
        p = d / f"{obj_id}.npz"
        if p.exists():
            return p
    return None


def events(obj_id: str, horizon_days: float = 100.0) -> np.ndarray:
    """(N, 15) event array truncated to the horizon; seeded, local file or fetched photometry."""
    snapshot = _SNAPSHOT.get()
    if snapshot is not None and snapshot[0] == obj_id:
        arr = snapshot[1]
        return arr[arr[:, 0] <= horizon_days]
    if obj_id in _SEEDED:
        arr = _SEEDED[obj_id][0]
        return arr[arr[:, 0] <= horizon_days]
    p = local_event_path(obj_id)
    if p is not None:
        return D.load_events(str(p), horizon_days=horizon_days)
    from flare.fetch import fetch_events
    arr = fetch_events(obj_id, horizon_days=horizon_days)
    if arr is None or not len(arr):
        raise ValueError(f"no photometry found for {obj_id}")
    return arr


@lru_cache(maxsize=1)
def _catalog():
    f = benchmark_file("bts_catalog.csv")
    if f is None:
        return None
    cat = pd.read_csv(f, low_memory=False).rename(columns={"ZTFID": "obj_id"})
    cat["peakmag"] = pd.to_numeric(cat.peakmag, errors="coerce")
    return cat.set_index("obj_id")


def _sex(v, is_ra):
    if isinstance(v, str) and ":" in v:
        p = [float(x) for x in v.split(":")]; s = -1 if v.strip().startswith("-") else 1
        return s * (abs(p[0]) + p[1] / 60 + (p[2] if len(p) > 2 else 0) / 3600) * (15 if is_ra else 1)
    return float(v)


def coords(obj_id: str):
    cat = _catalog()
    if cat is not None and obj_id in cat.index:
        r = cat.loc[obj_id]; return _sex(r.RA, True), _sex(r.Dec, False)
    from flare.fetch import coords as fc
    return fc(obj_id)


def catalog_row(obj_id: str) -> dict:
    cat = _catalog()
    if cat is None or obj_id not in cat.index:
        return {}
    r = cat.loc[obj_id]
    return dict(bts_type=None if str(r.get("type", "-")).strip() in ("-", "nan") else str(r["type"]),
                peakmag=float(r.peakmag) if np.isfinite(r.peakmag) else None,
                redshift=float(r.redshift) if "redshift" in r and pd.notna(r.redshift) else None)


@lru_cache(maxsize=1)
def _context_table():
    parts = []
    for name, cols in TABLES.items():
        f = benchmark_file(name)
        if f is not None:
            t = pd.read_csv(f)
            parts.append(t[["obj_id"] + [c for c in cols if c in t.columns]].drop_duplicates("obj_id").set_index("obj_id"))
    return pd.concat(parts, axis=1).groupby(level=0).first() if parts else None


def context(obj_id: str, peak_mag: float | None = None, live: bool = True) -> dict:
    """External context: seeded, the benchmark tables when the object is in them, else live catalogue queries."""
    snapshot = _SNAPSHOT.get()
    if snapshot is not None and snapshot[0] == obj_id:
        return {"_source": "provided", **copy.deepcopy(snapshot[2])}
    if obj_id in _SEEDED:
        return {"_source": "provided", **copy.deepcopy(_SEEDED[obj_id][1])}
    t = _context_table()
    if t is not None and obj_id in t.index:
        row = {k: float(v) for k, v in t.loc[obj_id].items() if isinstance(v, (int, float, np.floating)) and np.isfinite(v)}
        row["_source"] = "benchmark"; return row
    if not live:
        return {"_source": "none"}
    c = coords(obj_id)
    if not c:
        return {"_source": "none"}
    from flare.context import context_features
    out = context_features(c[0], c[1], peak_mag=peak_mag)
    out = {k: float(v) for k, v in out.items() if isinstance(v, (int, float, np.floating)) and np.isfinite(v)}
    out["_source"] = "live"; return out


def history(obj_id: str) -> dict:
    """The object's full photometric history from the broker archive (not truncated):
    detections, first and last epoch, activity span in days, per-band counts."""
    from flare.fetch import fetch_photometry
    rows = fetch_photometry(obj_id)
    if not rows:
        return dict(found=False)
    mjd = np.array([r[0] for r in rows], float); fid = np.array([int(r[1]) for r in rows])
    return dict(found=True, n_detections=int(len(rows)), first_mjd=round(float(mjd.min()), 3), last_mjd=round(float(mjd.max()), 3),
                activity_span_days=round(float(mjd.max() - mjd.min()), 1), n_per_band={b: int((fid == k).sum()) for k, b in ((1, "g"), (2, "r"), (3, "i"))})
