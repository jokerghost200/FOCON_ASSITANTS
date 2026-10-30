"""
FATI Feedback Engine (Moteur de Boucle de Rétroaction et d'Apprentissage Continu)
=================================================================================
Orchestre l'ensemble du cycle d'apprentissage après la clôture de chaque trade :
1. Enregistre le trade dans TradeJournal et ExperienceMemory
2. Lance la critique réflexive diagnostique (ReflectiveCritic)
3. Évalue la dérive de performance (DriftDetector)
4. Ajuste prudemment les pondérations adaptatives (AdaptiveWeightsEngine)
5. Notifie les leçons apprises de manière structurée
"""

from typing import Dict, Any, Optional
import logging
from src.learning.trade_journal import TradeJournal
from src.learning.experience_memory import ExperienceMemory
from src.learning.drift_detector import PerformanceDriftDetector
from src.learning.reflective_critic import ReflectiveCritic
from src.learning.adaptive_weights import AdaptiveWeightsEngine

logger = logging.getLogger(__name__)


class FeedbackEngine:
    """
    Pilote l'apprentissage post-trade autonome de FATI.
    """

    def __init__(
        self,
        journal: Optional[TradeJournal] = None,
        experience: Optional[ExperienceMemory] = None,
        drift_detector: Optional[PerformanceDriftDetector] = None,
        weights_engine: Optional[AdaptiveWeightsEngine] = None,
        critic: Optional[ReflectiveCritic] = None,
    ):
        self.journal = journal or TradeJournal()
        self.experience = experience or ExperienceMemory()
        self.drift_detector = drift_detector or PerformanceDriftDetector()
        self.weights_engine = weights_engine or AdaptiveWeightsEngine()
        self.critic = critic or ReflectiveCritic()

    def process_closed_trade(
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
        Traite un trade clôturé et exécute la boucle complète d'auto-évaluation.
        """
        # 1. Enregistrement dans le journal
        closed_rec = self.journal.record_closure(
            ticket=ticket,
            exit_price=exit_price,
            pnl_usd=pnl_usd,
            exit_reason=exit_reason,
            spread_cost=spread_cost,
            mfe=mfe,
            mae=mae,
        )

        if not closed_rec:
            # Fallback objet minimal si le trade n'était pas pré-enregistré
            closed_rec = {
                "ticket": ticket,
                "exit_price": exit_price,
                "pnl_usd": pnl_usd,
                "result_r": round(pnl_usd / 10.0, 3),
                "exit_reason": exit_reason,
            }

        # Protection anti-doublon idempotente : Si le trade était déjà clôturé, ne pas ré-appliquer l'apprentissage
        if closed_rec.get("already_closed"):
            logger.info(f"ℹ️ [FATI Feedback] Trade #{ticket} déjà clôturé dans TradeJournal. Ignoré pour éviter la double dérive/pondération.")
            return {"trade": closed_rec, "status": "ALREADY_PROCESSED"}

        # 2. Indexation dans la mémoire d'expérience
        self.experience.add_experience(closed_rec)

        # 3. Diagnostic critique
        critique = self.critic.analyze_trade(closed_rec, post_exit_candles=post_exit_candles)

        # 4. Ajustement des poids adaptatifs selon le résultat
        strategy = closed_rec.get("strategy", "GENERIC")
        is_win = pnl_usd > 0.0
        self.weights_engine.record_trade_outcome(setup_type=strategy, is_win=is_win, pnl=pnl_usd)

        # 5. Détection de dérive sur les trades récents
        recent_trades = self.journal.get_closed_trades(limit=20)
        drift_report = self.drift_detector.evaluate_drift(recent_trades)

        logger.info(
            f"🧠 [FATI Feedback] Trade #{ticket} analysé : PnL={pnl_usd:.2f}$, "
            f"Result={closed_rec.get('result_r', 0.0):.2f}R | "
            f"Diagnostic: {critique.get('summary', 'OK')} | Dérive: {drift_report.get('status')}"
        )

        return {
            "trade": closed_rec,
            "critique": critique,
            "drift": drift_report,
            "weights": self.weights_engine.get_weights(),
        }
