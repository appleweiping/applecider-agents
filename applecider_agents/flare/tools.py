"""FLARE-specific tools: the phase-matched classifier, the blind consensus anomaly
alarm, and an inline-photometry entry for clients that already hold the light curve."""
from __future__ import annotations
import math
import time
from contextvars import ContextVar
from contextlib import contextmanager
from anthropic.lib.tools import beta_tool
from ..core import config as C, data as DATA
from flare import data as D
from flare.skyportal import feature_frame, classify, phase_horizon

_CACHE: dict = {}
_RESULTS: dict = {}
_SESSION_CACHES = ContextVar("flare_snapshot_caches", default=None)
_WORK_OBSERVER = ContextVar("flare_work_observer", default=None)
BAND = {0: "g", 1: "r", 2: "i"}


def _f(x, nd=3):
    try:
        v = float(x)
        return None if not math.isfinite(v) else round(v, nd)
    except (TypeError, ValueError):
        return None


@contextmanager
def scoped_cache(observer=None):
    """Own science results/cost instrumentation within one supplied snapshot."""
    token = _SESSION_CACHES.set(({}, {}))
    work_token = _WORK_OBSERVER.set(observer)
    try:
        yield
    finally:
        _WORK_OBSERVER.reset(work_token)
        _SESSION_CACHES.reset(token)


def _caches():
    scoped = _SESSION_CACHES.get()
    return scoped if scoped is not None else (_CACHE, _RESULTS)


def cached_result(obj_id, horizon_days):
    return _cache_key(obj_id, horizon_days) in _caches()[1]


def _work(backend, fn, *args, **kwargs):
    observer = _WORK_OBSERVER.get()
    if observer is None:
        return fn(*args, **kwargs)
    started = time.perf_counter()
    record = {"backend": backend}
    if backend == "classify":
        from flare.skyportal import _CLASSIFIERS
        record["phase_classifier_cache"] = "warm" if kwargs.get("horizon") in _CLASSIFIERS else "cold"
    try:
        return fn(*args, **kwargs)
    finally:
        record["wall_seconds"] = time.perf_counter() - started
        observer(record)

def _cache_key(obj_id: str, horizon_days: float):
    revision = DATA.revision(obj_id)
    # Retain horizons for the current snapshot only; replaced snapshots cannot grow stale caches.
    for cache in _caches():
        for old in list(cache):
            if old[0] == obj_id and (len(old) < 3 or old[2] != revision):
                del cache[old]
    return obj_id, float(horizon_days), revision


def prepared(obj_id: str, horizon_days: float = 100.0):
    """Events, context, feature frame and observed peak for one object (cached per process)."""
    key = _cache_key(obj_id, horizon_days)
    cache = _caches()[0]
    if key not in cache:
        ev = DATA.events(obj_id, horizon_days)
        peak0 = float(23.9 - 2.5 * ev[:, D.COL["logflux"]].max())
        ctx = DATA.context(obj_id, peak_mag=peak0)
        X, peak, _ = _work("feature_frame", feature_frame, ev, {k: v for k, v in ctx.items() if not k.startswith("_")}, None)
        cache[key] = (ev, ctx, X, peak)
    return cache[key]


def _flare(obj_id: str, horizon_days: float = 100.0) -> dict:
    """FLARE's classification and anomaly alarm (flare.skyportal.classify) from the model matched to
    the span of the detections inside the horizon (flare.skyportal.phase_horizon); cached per process."""
    key = _cache_key(obj_id, horizon_days)
    cache = _caches()[1]
    if key not in cache:
        ev, _, X, _ = prepared(obj_id, horizon_days)
        span = float(ev[:, D.COL["dt"]].max() - ev[:, D.COL["dt"]].min())
        cache[key] = _work("classify", classify, X, C.BASE_RATE, horizon=phase_horizon(span))
    return cache[key]


@beta_tool
def flare_classify(obj_id: str, horizon_days: float = 100.0) -> dict:
    """Run FLARE (six classes: SN_Ia, SN_CC, SLSN, AGN, TDE, CV) on the object's
    light curve plus its catalogue context, with the model matched to its phase:
    the one trained at day 5, 10 or 20 (bts6_h5/h10/h20, chosen by the span of
    the detections) or the full-curve bts6. Returns the model, class
    probabilities with their Venn-Abers intervals, the argmax, Mondrian
    conformal p-values, the 90% prediction set, credibility (largest p-value;
    low means the object matches no class well), confidence, whether the set
    excludes its own argmax, and the measured coverage of the predicted class."""
    try:
        _, ctx, _, _ = prepared(obj_id, horizon_days); cl = _flare(obj_id, horizon_days)
        return dict(obj_id=obj_id, model=cl["model"], model_horizon_days=cl["model_horizon_days"], predicted=cl["predicted"],
                    probabilities={k: _f(v, 4) for k, v in cl["probabilities"].items()}, probability_intervals=cl["probability_intervals"],
                    prediction_set=cl["prediction_set"], alpha=cl["alpha"], p_values={k: _f(v, 4) for k, v in cl["p_values"].items()},
                    credibility=_f(cl["credibility"], 4), confidence=_f(cl["confidence"], 4),
                    argmax_excluded_from_set=cl["argmax_excluded_from_set"],
                    coverage_of_predicted=cl["coverage_of_predicted"], coverage_of_predicted_err=cl["coverage_of_predicted_err"],
                    context_used=sorted(k for k in ctx if not k.startswith("_")), context_source=ctx.get("_source"),
                    note="Probabilities are out-of-fold calibrated on the BTS benchmark; the set carries the 90% per-class guarantee. Early models give wider sets by design.")
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}")


ALARM_KEYS = ("alarm_score", "consensus", "alarm_p", "alarms", "alarm_far", "calibrated", "calibration")


@beta_tool
def flare_anomaly(obj_id: str, horizon_days: float = 100.0) -> dict:
    """Blind anomaly alarm from FLARE, from the same phase-matched model as
    flare_classify: the consensus of 14 LUNA scorers (alarm_score 'consensus';
    'energy' only when LUNA is missing), its conformal p-value alarm_p, the
    decision at 1%, 5% and 10% false-alarm rates (alarms) and alarm_far, the
    smallest rate that flags the object (None when none does). calibrated says
    whether the alarm is calibrated at the model's phase. The energy, its
    percentile, the novelty p-value, the likelihood ratio and P(anomaly) at the
    configured base rate are returned as diagnostics."""
    try:
        cl = _flare(obj_id, horizon_days); a = cl["anomaly"]
        out = dict(obj_id=obj_id, model=cl["model"], **{k: a.get(k) for k in ALARM_KEYS})
        out.update({k: (_f(v, 4) if isinstance(v, float) else v) for k, v in a.items() if k not in ALARM_KEYS})
        out["note"] = ("The 1% alarm is the operating point and counts only from 8 detections; the energy fields are diagnostics. "
                       "The base rate is the benchmark's (3.3%); a raw alert stream is far poorer in rarities, so quote the likelihood ratio when the prior is unknown.")
        return out
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}")




def flare_analyze_photometry(mjd: list[float], filter: list[str], mag: list[float], magerr: list[float], obj_id: str = "obj",
                             redshift: float | None = None, ra: float | None = None, dec: float | None = None,
                             horizon_days: float = 100.0, context: str = "live") -> dict:
    """Run FLARE directly on photometry the caller already holds (e.g. SkyPortal's get_photometry): lists of mjd, filter (ztfg/ztfr/ztfi), mag, magerr; optional redshift, ra, dec. Returns the FLARE analysis-service result (phase-matched classification, consensus anomaly alarm, context, rule verdict, results.forecast, SkyPortal-shaped annotations and classification, and model_lightcurve, the forecast g and r curves, when a forecast is made)."""
    from flare.skyportal import analyze
    fid = {"ztfg": 1, "ztfr": 2, "ztfi": 3, "g": 1, "r": 2, "i": 3}
    rows = sorted((float(t), fid[f.lower()], float(m), float(e)) for t, f, m, e in zip(mjd, filter, mag, magerr) if f.lower() in fid)
    params = dict(horizon_days=horizon_days, context=context, plot=False)
    if ra is not None and dec is not None:
        params.update(ra=ra, dec=dec)
    out = analyze(rows, redshift, params, resource_id=obj_id, work_dir="/tmp"); out.pop("plot_files", None)
    import json; return json.loads(json.dumps(out, default=float))


TOOLS = [flare_classify, flare_anomaly]
