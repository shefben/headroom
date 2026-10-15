"""Local intelligence: JevK5 setup, managed llama-server lifecycle and the DecisionAdvisor.

JevK5 is an optional advisory classifier. Nothing here can establish facts,
execute commands or authorize file changes; every failure path returns "no
advice" and deterministic behavior continues.
"""

from headroom.intelligence.advisor import Advice, DecisionAdvisor, RequestBudget
from headroom.intelligence.manager import IntelligenceManager
from headroom.intelligence.settings import IntelligenceSettings, ServicePolicy

__all__ = [
    "Advice",
    "DecisionAdvisor",
    "IntelligenceManager",
    "IntelligenceSettings",
    "RequestBudget",
    "ServicePolicy",
]
