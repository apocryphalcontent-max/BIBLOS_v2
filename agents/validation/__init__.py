"""
BIBLOS v2 - Validation Agents

Agents for validating extraction quality:
- ELENKTIKOS: Cross-agent consistency checker
- KRITIKOS: Quality scoring and critique
- HARMONIZER: Result harmonization
- PROSECUTOR: Challenge agent
- WITNESS: Defense agent
"""

from agents.validation.elenktikos import ElenktikosAgent
from agents.validation.kritikos import KritikosAgent
from agents.validation.harmonizer import HarmonizerAgent
from agents.validation.prosecutor import ProsecutorAgent
from agents.validation.witness import WitnessAgent

__all__ = [
    "ElenktikosAgent",
    "KritikosAgent",
    "HarmonizerAgent",
    "ProsecutorAgent",
    "WitnessAgent",
]
