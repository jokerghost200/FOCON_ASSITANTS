"""
Package d'apprentissage réflexif et de mémoire évolutive FATI.
Fournit :
- TradeMemoryEngine / TradeRecord : Mémoire épisodique existante
- ReflectiveCritic : Module d'auto-critique post-mortem
- AdaptiveWeightsEngine : Calibrage dynamique et borné des poids
- TradeJournal : Journal persistant SQLite des décisions et résultats
- ExperienceMemory : Mémoire structurée des expériences et patterns
- KnowledgeBase : Base de connaissances et règles certifiées
- PerformanceAnalyzer : Analyseur de performance quantitative (Sharpe, Expectancy R, PF)
- PerformanceDriftDetector : Surveillance continue de dérive statistique
- FeedbackEngine : Moteur d'orchestration de la boucle de rétroaction post-trade
"""
from src.learning.trade_memory import TradeMemoryEngine, TradeRecord
from src.learning.reflective_critic import ReflectiveCritic
from src.learning.adaptive_weights import AdaptiveWeightsEngine
from src.learning.trade_journal import TradeJournal
from src.learning.experience_memory import ExperienceMemory
from src.learning.knowledge_base import KnowledgeBase
from src.learning.performance_analyzer import PerformanceAnalyzer
from src.learning.drift_detector import PerformanceDriftDetector, DriftStatus
from src.learning.feedback_engine import FeedbackEngine

__all__ = [
    "TradeMemoryEngine",
    "TradeRecord",
    "ReflectiveCritic",
    "AdaptiveWeightsEngine",
    "TradeJournal",
    "ExperienceMemory",
    "KnowledgeBase",
    "PerformanceAnalyzer",
    "PerformanceDriftDetector",
    "DriftStatus",
    "FeedbackEngine",
]
