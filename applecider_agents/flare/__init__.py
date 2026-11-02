"""FLARE model agent: photometric classification, conformal sets, blind consensus anomaly alarm."""
from ..core.registry import ModelAgent
from .tools import TOOLS, flare_analyze_photometry
from .triage import SYSTEM, READING_GUIDE, mock_triage
from .observations import flare_partial_triage

AGENT = ModelAgent(name="flare", description="photometric classifier (SN Ia, SN CC, SLSN, AGN, TDE, CV) with 90% conformal sets and a blind consensus anomaly alarm on ZTF light curves plus alert-time catalogue context, using the model matched to the light curve's phase.",
                   tools=TOOLS, triage=mock_triage, system_prompt=SYSTEM, reading_guide=READING_GUIDE, mcp_extra=[flare_analyze_photometry, flare_partial_triage]).register()
