"""
Package d'apprentissage réflexif et de mémoire évolutive pour le trading assistant.
Fournit :
- TradeMemoryEngine : Journalisation et mémoire épisodique des trades avec contexte de marché.
- ReflectiveCritic : Module d'auto-critique post-mortem des trades gagnants et perdants.
- AdaptiveWeightsEngine : Calibrage dynamique et sécurisé des poids de décision quantitative.
"""
from src.learning.trade_memory import TradeMemoryEngine, TradeRecord
from src.learning.reflective_critic import ReflectiveCritic
from src.learning.adaptive_weights import AdaptiveWeightsEngine

__all__ = [
    "TradeMemoryEngine",
    "TradeRecord",
    "ReflectiveCritic",
    "AdaptiveWeightsEngine",
]
