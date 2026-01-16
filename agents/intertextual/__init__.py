"""
BIBLOS v2 - Intertextual Extraction Agents

Agents for intertextual analysis of biblical texts:
- SYNDESMOS: Cross-reference connection analysis
- HARMONIKOS: Parallel passage harmonization
- ALLOGRAPHOS: Quotation and allusion detection
- PARADEIGMA: Example and precedent identification
- TOPOS: Common topic/motif analysis
"""

from agents.intertextual.syndesmos import SyndesmosAgent
from agents.intertextual.harmonikos import HarmonikosAgent
from agents.intertextual.allographos import AllographosAgent
from agents.intertextual.paradeigma import ParadeigmaAgent
from agents.intertextual.topos import ToposAgent

__all__ = [
    "SyndesmosAgent",
    "HarmonikosAgent",
    "AllographosAgent",
    "ParadeigmaAgent",
    "ToposAgent",
]
