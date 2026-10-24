"""
FOCON Autonomous Trading Intelligence (FATI)
============================================
Cerveau supérieur autonome à 5 piliers :
1. OBSERVER  : MarketRegimeDetector (tendance, range, volatilité, illiquidité)
2. RÉFLÉCHIR : DecisionEngine (probabilités calibrées, espérance E(R), NO-TRADE)
3. APPRENDRE : TradeJournal, ExperienceMemory, KnowledgeBase, FeedbackEngine
4. EXPÉRIMENTER : HypothesisGenerator, ExperimentRunner, WalkForwardValidator
5. SÉCURISER : SafetyRiskGuard, PromotionGate, RollbackManager, ModelRegistry
"""

from typing import Dict, Any, List, Optional
import logging

from src.core.market_regime import MarketRegimeDetector
from src.core.decision_engine import DecisionEngine
from src.learning.trade_journal import TradeJournal
from src.learning.experience_memory import ExperienceMemory
from src.learning.knowledge_base import KnowledgeBase
from src.learning.performance_analyzer import PerformanceAnalyzer
from src.learning.drift_detector import PerformanceDriftDetector
from src.learning.feedback_engine import FeedbackEngine
from src.research.hypothesis_generator import HypothesisGenerator
from src.research.experiment_runner import ExperimentRunner
from src.research.walk_forward import WalkForwardValidator
from src.research.model_registry import ModelRegistry
from src.safety.risk_guard import SafetyRiskGuard
from src.safety.promotion_gate import PromotionGate
from src.safety.rollback import RollbackManager
from src.services.learning_scheduler import LearningScheduler
from src.services.health_monitor import SystemHealthMonitor

logger = logging.getLogger(__name__)


class FATIBrain:
    """
    Système unifié d'Intelligence Autonome de Trading (FATI).
    Fonctionne comme couche supérieure sans jamais perturber les composants existants.
    """

    def __init__(
        self,
        mt5_connector=None,
        local_brain=None,
        weights_engine=None,
        critic=None,
    ):
        # 1. Piliers d'Observation & Décision
        self.regime_detector = MarketRegimeDetector()
        self.decision_engine = DecisionEngine(local_brain=local_brain, regime_detector=self.regime_detector)

        # 2. Piliers d'Apprentissage & Mémoire
        self.journal = TradeJournal()
        self.experience = ExperienceMemory()
        self.knowledge = KnowledgeBase()
        self.analyzer = PerformanceAnalyzer()
        self.drift_detector = PerformanceDriftDetector()
        self.feedback = FeedbackEngine(
            journal=self.journal,
            experience=self.experience,
            drift_detector=self.drift_detector,
            weights_engine=weights_engine,
            critic=critic,
        )

        # 3. Piliers de Recherche & Expérimentation
        self.hypotheses = HypothesisGenerator()
        self.experiments = ExperimentRunner()
        self.walk_forward = WalkForwardValidator()
        self.registry = ModelRegistry()

        # 4. Piliers de Sécurité & Promotion
        self.risk_guard = SafetyRiskGuard()
        self.promotion_gate = PromotionGate()
        self.rollback = RollbackManager(registry=self.registry, weights_engine=weights_engine)

        # 5. Services d'infrastructure
        self.scheduler = LearningScheduler(interval_seconds=1800)
        self.health = SystemHealthMonitor(
            mt5_connector=mt5_connector,
            journal=self.journal,
            drift_detector=self.drift_detector,
            safety_guard=self.risk_guard,
            registry=self.registry,
        )

        # Enregistrement des tâches périodiques du planificateur
        self.scheduler.register_task("audit_drift", self._scheduled_drift_audit)
        self.scheduler.register_task("discover_toxic_patterns", self._scheduled_pattern_discovery)

        logger.info("🧠 [FATI] Cerveau Autonome initialisé avec ses 5 piliers complets.")

    def observe_and_decide(
        self,
        symbol: str,
        timeframe: str,
        tech_features: Dict[str, Any],
        amd_context: Optional[Dict[str, Any]] = None,
        boomerang_data: Optional[Dict[str, Any]] = None,
        order_blocks: Optional[List[Any]] = None,
        crt_setup: Optional[Any] = None,
        htf_trend: str = "Unknown",
        spread_price: float = 0.0,
        direction_filter: str = "BOTH",
    ) -> Dict[str, Any]:
        """
        Cycle Pilier 1 & 2 : Observe et Décide.
        Journalise la décision pour la postérité statistique.
        """
        decision = self.decision_engine.evaluate(
            symbol=symbol,
            timeframe=timeframe,
            tech_features=tech_features,
            amd_context=amd_context,
            boomerang_data=boomerang_data,
            order_blocks=order_blocks,
            crt_setup=crt_setup,
            htf_trend=htf_trend,
            spread_price=spread_price,
            direction_filter=direction_filter,
        )

        # Enregistrement systématique de la décision dans le journal
        decision_id = self.journal.log_decision(decision, is_executed=(decision.get("action") in ("BUY", "SELL")))
        decision["fati_decision_id"] = decision_id
        return decision

    def learn_from_trade(
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
        Cycle Pilier 3 : Apprend d'un trade clôturé.
        Si une dérive critique est détectée, active automatiquement la mise en quarantaine / rollback.
        """
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

        drift = report.get("drift", {})
        if drift.get("status") == "CRITICAL_QUARANTINE":
            logger.critical("🚨 [FATI] Dérive critique détectée en production. Déclenchement du rollback...")
            self.rollback.trigger_rollback(incident_reason=drift.get("reason", "Dérive d'espérance critique"))

        return report

    def _scheduled_drift_audit(self):
        """Tâche d'arrière-plan : surveillance de dérive de performance."""
        recent = self.journal.get_closed_trades(limit=25)
        self.drift_detector.evaluate_drift(recent)

    def _scheduled_pattern_discovery(self):
        """Tâche d'arrière-plan : détection de patterns toxiques et génération d'hypothèses."""
        toxic = self.experience.detect_toxic_patterns(min_sample=5, max_win_rate=0.30)
        if toxic:
            hypotheses = self.hypotheses.generate_from_toxic_patterns(toxic)
            logger.info(f"💡 [FATI Research] {len(hypotheses)} hypothèses d'amélioration générées.")
