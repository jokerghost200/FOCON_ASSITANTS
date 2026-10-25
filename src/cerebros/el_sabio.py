# -*- coding: utf-8 -*-
"""
CEREBRO 3 : EL SABIO (Pilier 3 - APPRENDRE & MÉMORISER)
========================================================
Nom Espagnol : El Sabio ("Le Sage / L'Érudit") - Alias : La Memoria
Rôle : Mémorise chaque expérience, apprend des erreurs, adapte dynamiquement les pondérations
et détecte toute dérive statistique du comportement du marché.

Composants fédérés :
- TradeJournal : Journal haute précision enregistrant chaque décision et chaque trade
- ExperienceMemory : Base de mémoire épisodique extrayant les leçons et patterns toxiques
- KnowledgeBase : Base de connaissances certifiées et règles apprises
- PerformanceAnalyzer : Analyse statistique approfondie de performance
- PerformanceDriftDetector : Sentinelle détectant la baisse d'efficacité d'une stratégie
- FeedbackEngine : Boucle de rétroaction post-trade
- AdaptiveWeightsEngine : Mise à jour adaptative des poids de confluences
- ReflectiveCritic : Auto-critique réflexive
"""

import logging
from typing import Dict, Any, List, Optional

from src.learning.trade_journal import TradeJournal
from src.learning.experience_memory import ExperienceMemory
from src.learning.knowledge_base import KnowledgeBase
from src.learning.performance_analyzer import PerformanceAnalyzer
from src.learning.drift_detector import PerformanceDriftDetector
from src.learning.feedback_engine import FeedbackEngine
from src.learning.adaptive_weights import AdaptiveWeightsEngine
from src.learning.reflective_critic import ReflectiveCritic

logger = logging.getLogger(__name__)


class ElSabio:
    """
    Cerveau Apprenant & Mémorial (El Sabio)
    Garantit que le robot développe une sagesse statistique continue sans jamais répéter deux fois
    la même erreur sur le marché.
    """

    def __init__(
        self,
        journal: Optional[TradeJournal] = None,
        experience: Optional[ExperienceMemory] = None,
        knowledge: Optional[KnowledgeBase] = None,
        analyzer: Optional[PerformanceAnalyzer] = None,
        drift_detector: Optional[PerformanceDriftDetector] = None,
        weights_engine: Optional[AdaptiveWeightsEngine] = None,
        critic: Optional[ReflectiveCritic] = None,
    ):
        self.journal = journal or TradeJournal()
        self.experience = experience or ExperienceMemory()
        self.knowledge = knowledge or KnowledgeBase()
        self.analyzer = analyzer or PerformanceAnalyzer()
        self.drift_detector = drift_detector or PerformanceDriftDetector()
        self.weights_engine = weights_engine or AdaptiveWeightsEngine()
        self.critic = critic or ReflectiveCritic()

        self.feedback = FeedbackEngine(
            journal=self.journal,
            experience=self.experience,
            drift_detector=self.drift_detector,
            weights_engine=self.weights_engine,
            critic=self.critic,
        )

    def registrar_decision(self, decision: Dict[str, Any], is_executed: bool = False) -> str:
        """Enregistre une décision stratégique dans le journal pour postérité statistique."""
        return self.journal.log_decision(decision, is_executed=is_executed)

    def aprender_de_trade(
        self,
        ticket: int,
        exit_price: float,
        pnl_usd: float,
        exit_reason: str,
        spread_cost: float = 0.0,
        mfe: float = 0.0,
        mae: float = 0.0,
        post_exit_candles: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Traite un trade clôturé : enregistrement du PnL, analyse MFE/MAE,
        critique réflexive et adaptation automatique des poids.
        """
        logger.info(f"🧬 [El Sabio] Intégration du trade #{ticket} (PnL: {pnl_usd:.2f}$, Sortie: {exit_reason})")
        report = self.feedback.process_closed_trade(
            ticket=ticket,
            exit_price=exit_price,
            pnl_usd=pnl_usd,
            exit_reason=exit_reason,
            spread_cost=spread_cost,
            mfe=mfe,
            mae=mae,
            post_exit_candles=post_exit_candles,
        )
        return report

    def obtener_lecciones_recientes(self, symbol: Optional[str] = None, limit: int = 5) -> List[str]:
        """Fournit au cerveau stratège les leçons historiques récentes sur ce symbole."""
        return self.experience.get_lessons_for_symbol(symbol=symbol, limit=limit) if hasattr(self.experience, "get_lessons_for_symbol") else []

    def detectar_patrones_toxicos(self, min_sample: int = 5, max_win_rate: float = 0.30) -> List[Dict[str, Any]]:
        """Identifie les contextes récurrents où le robot perd de l'argent afin de les proscrire."""
        return self.experience.detect_toxic_patterns(min_sample=min_sample, max_win_rate=max_win_rate)
