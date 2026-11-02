"""Tools about the object itself, shared by every model agent: light curve, catalogue
context, the full photometric history, the BTS catalogue entry, TNS. Never raise to the model: failures
come back as {"error": ...}."""
from __future__ import annotations
import json, math, urllib.request, urllib.parse
import numpy as np
from anthropic.lib.tools import beta_tool
from . import config as C, data as DATA
from flare import data as D
from flare.context import m_pseudo

BAND = {0: "g", 1: "r", 2: "i"}


def _f(x, nd=3):
    try:
        v = float(x); return None if not math.isfinite(v) else round(v, nd)
    except (TypeError, ValueError):
        return None


def _events_and_context(obj_id, horizon_days=100.0):
    ev = DATA.events(obj_id, horizon_days)
    peak = float(23.9 - 2.5 * ev[:, D.COL["logflux"]].max())
    return ev, peak, DATA.context(obj_id, peak_mag=peak)


@beta_tool
def get_photometry(obj_id: str, horizon_days: float = 100.0) -> dict:
    """Light-curve summary for a ZTF object: detections per band, time span, peak and
    latest magnitudes, rise and fade, and a down-sampled list of (day, band, mag, err).
    Days count from the first detection; horizon_days shows only the first N days."""
    try:
        ev, peak, _ = _events_and_context(obj_id, horizon_days); bands = D.reconstruct_bands(ev)
        out = dict(obj_id=obj_id, horizon_days=horizon_days, n_detections=int(len(ev)), span_days=_f(ev[:, 0].max(), 1), peak_mag_observed=_f(peak, 2), bands={})
        for b, lc in bands.items():
            mag = 23.9 - 2.5 * np.asarray(lc.logflux); i = int(np.argmin(mag))
            out["bands"][b] = dict(n=int(len(mag)), peak_mag=_f(mag[i], 2), peak_day=_f(lc.t[i], 1), first_mag=_f(mag[0], 2), last_mag=_f(mag[-1], 2), last_day=_f(lc.t[-1], 1), rise_days=_f(lc.t[i] - lc.t[0], 1), fade_after_peak_mag=_f(mag[-1] - mag[i], 2))
        step = max(1, len(ev) // 30)
        out["points"] = [[_f(r[0], 1), BAND.get(int(r[D.COL["band_id"]]), "?"), _f(23.9 - 2.5 * r[D.COL["logflux"]], 2), _f(2.5 * r[D.COL["logflux_err"]], 2)] for r in ev[::step]]
        return out
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}")


@beta_tool
def get_context(obj_id: str, horizon_days: float = 100.0) -> dict:
    """Catalogue context at the transient position, all available before any spectrum:
    Pan-STARRS host (offset, magnitudes, colours), host photometric redshift and the
    implied pseudo-absolute magnitude, Gaia point source (parallax / proper-motion
    significance), WISE W1-W2, pre-outburst Pan-STARRS counterpart within 1.5 arcsec.
    Plain flags: nuclear, hostless, stellar astrometry, AGN colour, far host.
    horizon_days limits the observed peak used for luminosity; default100 preserves legacy calls."""
    try:
        ev, peak, ctx = _events_and_context(obj_id, horizon_days); g = lambda k: ctx.get(k); z = g("z_phot")
        out = dict(obj_id=obj_id, source=ctx.get("_source"),
                   host=dict(offset_arcsec=_f(g("host_sep"), 2), r_mag=_f(g("host_r"), 2), i_mag=_f(g("host_i"), 2), g_r=_f(g("host_gr"), 2), extended_neighbours_30arcsec=_f(g("n_ext_30"), 0)),
                   redshift=dict(z_phot=_f(z, 3), z_spec_host=_f(g("z_spec"), 4), M_pseudo=_f(m_pseudo(peak, z), 2) if z is not None and z > 0 else None,
                                 note="M_pseudo uses the observed peak and the HOST photo-z; on the labelled benchmark M_pseudo < -21 alone marks a true SLSN only 1% of the time"),
                   gaia=dict(point_source_within_2arcsec=g("gaia_sep") is not None, G=_f(g("gaia_g"), 2), parallax_over_error=_f(g("parallax_over_error"), 1), pm_over_error=_f(g("pm_over_error"), 1), ruwe=_f(g("ruwe"), 2)),
                   wise=dict(match_within_6arcsec=g("wise_sep") is not None, w1_w2=_f(g("w1w2"), 2)),
                   ps1_counterpart=dict(within_1p5arcsec=g("pos_sep") is not None, r_mag=_f(g("pos_r"), 2), point_like=(g("pos_star") == 1.0) if g("pos_star") is not None else None, n_detections=_f(g("pos_ndet"), 0)))
        out["flags"] = dict(hostless=g("host_sep") is None, nuclear=(g("host_sep") is not None and g("host_sep") < 0.5),
                            offset_from_host_gt_10arcsec=(g("host_sep") is not None and g("host_sep") > 10),
                            stellar_astrometry=bool((g("parallax_over_error") or 0) > 3 or (g("pm_over_error") or 0) > 3),
                            agn_mid_ir_colour=bool((g("w1w2") or -9) > 0.8),
                            superluminous_if_host_is_right=bool(z is not None and z > 0 and m_pseudo(peak, z) < -21))
        return out
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}")


@beta_tool
def light_curve_history(obj_id: str) -> dict:
    """The object's complete photometric history from the broker archive, not limited to
    the 100-day window the classifier sees: detections, first and last epoch, activity span
    in days. A span of years means a persistent variable (AGN, stellar), not a transient."""
    try:
        return dict(obj_id=obj_id, **DATA.history(obj_id))
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}")


@beta_tool
def catalog_lookup(obj_id: str) -> dict:
    """ZTF Bright Transient Survey catalogue entry: spectroscopic type if already
    classified (then it is not a discovery), catalogue peak magnitude, redshift, coordinates."""
    try:
        r = DATA.catalog_row(obj_id); c = DATA.coords(obj_id)
        return dict(obj_id=obj_id, in_bts_catalog=bool(r), spectroscopic_type=r.get("bts_type"), already_classified=bool(r.get("bts_type")),
                    peakmag=r.get("peakmag"), redshift=r.get("redshift"), ra=_f(c[0], 5) if c else None, dec=_f(c[1], 5) if c else None)
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}")


@beta_tool
def tns_lookup(obj_id: str) -> dict:
    """Transient Name Server reports and classification within 3 arcsec (needs TNS bot
    credentials on the server; says so when absent)."""
    if not (C.TNS_BOT_ID and C.TNS_BOT_NAME and C.TNS_API_KEY):
        return dict(obj_id=obj_id, available=False, reason="TNS credentials not configured (TNS_BOT_ID, TNS_BOT_NAME, TNS_API_KEY)")
    try:
        c = DATA.coords(obj_id)
        if not c:
            return dict(obj_id=obj_id, available=False, reason="no coordinates")
        marker = 'tns_marker{"tns_id":%s,"type":"bot","name":"%s"}' % (C.TNS_BOT_ID, C.TNS_BOT_NAME)
        body = urllib.parse.urlencode({"api_key": C.TNS_API_KEY, "data": json.dumps(dict(ra=str(c[0]), dec=str(c[1]), radius="3", units="arcsec", internal_name=obj_id))}).encode()
        with urllib.request.urlopen(urllib.request.Request("https://www.wis-tns.org/api/get/search", data=body, headers={"User-Agent": marker}), timeout=30) as r:
            res = json.load(r)
        hits = res.get("data", {}).get("reply", []) if isinstance(res.get("data"), dict) else res.get("data", [])
        out = []
        for h in hits[:5]:
            body = urllib.parse.urlencode({"api_key": C.TNS_API_KEY, "data": json.dumps(dict(objname=h.get("objname"), photometry="0", spectra="0"))}).encode()
            with urllib.request.urlopen(urllib.request.Request("https://www.wis-tns.org/api/get/object", data=body, headers={"User-Agent": marker}), timeout=30) as r:
                o = json.load(r).get("data", {}).get("reply", {})
            out.append(dict(name=f"{h.get('prefix', '')} {h.get('objname')}", object_type=o.get("object_type", {}).get("name"), redshift=o.get("redshift"), discovery_date=o.get("discoverydate"), reporting_group=o.get("reporting_group", {}).get("group_name")))
        return dict(obj_id=obj_id, available=True, matches=out)
    except Exception as e:
        return dict(obj_id=obj_id, available=False, reason=f"{type(e).__name__}: {e}")


OBJECT_TOOLS = [get_photometry, get_context, light_curve_history, catalog_lookup, tns_lookup]
