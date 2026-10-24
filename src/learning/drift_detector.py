"""
FATI Drift Detector (Détecteur de Dérive de Marché et Dégradation de Performance)
================================================================================
Surveille en continu l'alignement entre les performances attendues et réelles :
- Baisse anormale de l'espérance de gain en R sur fenêtre glissante (Concept Drift)
- Détection des séries de pertes anormales (Drawdown drift)
- Déclenchement automatique de mises en quarantaine ou réduction de voilure (Safety De-risking)
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DriftStatus:
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL_QUARANTINE = "CRITICAL_QUARANTINE"


class PerformanceDriftDetector:
    """
    Surveille la dégradation des performances pour éviter de brûler du capital
    lorsque les conditions de marché rendent une stratégie obsolète.
    """

    def __init__(
        self,
        window_size: int = 15,
        min_expectancy_r: float = 0.05,
        max_consecutive_losses: int = 4,
        max_drawdown_limit_usd: float = 50.0,
        min_trades_to_evaluate: int = 5,
    ):
        self.window_size = window_size
        self.min_expectancy_r = min_expectancy_r
        self.max_consecutive_losses = max_consecutive_losses
        self.max_drawdown_limit_usd = max_drawdown_limit_usd
        self.min_trades_to_evaluate = min_trades_to_evaluate
        self.is_quarantined = False

    def evaluate_drift(self, recent_trades: List[Dict[str, Any]], baseline_expectancy: float = 0.35) -> Dict[str, Any]:
        """
        Évalue si les trades récents présentent une dérive statistique négative.
        """
        if len(recent_trades) < self.min_trades_to_evaluate:
            return {
                "status": DriftStatus.HEALTHY,
                "reason": "Échantillon trop récent pour évaluer une dérive.",
                "action_recommended": "CONTINUE",
                "risk_multiplier": 1.0,
            }

        window = recent_trades[: self.window_size]
        r_list = [float(t.get("result_r", 0.0) or 0.0) for t in window]
        current_exp_r = sum(r_list) / len(r_list)

        # 1. Analyse des pertes consécutives
        consecutive_losses = 0
        for t in window:
            if float(t.get("pnl_usd", 0.0) or 0.0) < 0:
                consecutive_losses += 1
            else:
                break

        # 2. Détection de défaillance critique
        if consecutive_losses >= self.max_consecutive_losses and current_exp_r < 0.0:
            self.is_quarantined = True
            logger.warning(
                f"🚨 [DriftDetector] Alerte CRITIQUE : {consecutive_losses} pertes consécutives, "
                f"espérance glissante négative ({current_exp_r:.2f} R)."
            )
            return {
                "status": DriftStatus.CRITICAL_QUARANTINE,
                "reason": f"Série de {consecutive_losses} pertes consécutives avec espérance glissante négative ({current_exp_r:.2f} R).",
                "action_recommended": "QUARANTINE_STRATEGY",
                "risk_multiplier": 0.0,
                "current_expectancy_r": current_exp_r,
                "consecutive_losses": consecutive_losses,
            }

        # 3. Détection d'avertissement / dégradation progressive
        if current_exp_r < self.min_expectancy_r and len(window) >= 10:
            logger.info(
                f"⚠️ [DriftDetector] Dégradation observée : espérance à {current_exp_r:.2f} R "
                f"(vs baseline {baseline_expectancy:.2f} R). Réduction du risque préventive à 50%."
            )
            return {
                "status": DriftStatus.WARNING,
                "reason": f"Espérance glissante ({current_exp_r:.2f} R) en baisse sous le seuil d'avantage statistique.",
                "action_recommended": "REDUCE_RISK_50_PCT",
                "risk_multiplier": 0.5,
                "current_expectancy_r": current_exp_r,
                "consecutive_losses": consecutive_losses,
            }

        return {
            "status": DriftStatus.HEALTHY,
            "reason": f"Performances stables (Espérance glissante: {current_exp_r:.2f} R).",
            "action_recommended": "CONTINUE",
            "risk_multiplier": 1.0,
            "current_expectancy_r": current_exp_r,
            "consecutive_losses": consecutive_losses,
        }
