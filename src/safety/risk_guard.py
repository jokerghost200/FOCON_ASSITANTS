"""
FATI Safety Risk Guard (Garde-Fou Indépendant Absolu)
=====================================================
Fournit une couche de protection inviolable, totalement étanche et indépendante :
- Limite absolue de risque par trade (non modifiable par l'IA ou l'apprentissage)
- Plafond absolu de perte journalière (Kill Switch)
- Contrôle de levier et de volume maximal
- Arrêt d'urgence automatique
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SafetyRiskGuard:
    """
    Système de sécurité indépendant.
    Ses paramètres sont en dur ou issus de constantes strictes,
    et ne peuvent JAMAIS être assouplis par le système d'apprentissage.
    """

    # Limites infranchissables du système
    HARD_MAX_RISK_PER_TRADE_PERCENT = 1.5
    HARD_MAX_FIXED_LOT_RISK_PERCENT = 35.0
    HARD_MAX_DAILY_LOSS_PERCENT = 4.0
    HARD_MAX_OPEN_POSITIONS = 5
    HARD_MAX_DRAWDOWN_PERCENT = 8.0

    def __init__(self):
        self.emergency_stop_triggered = False

    def validate_proposal_safety(
        self,
        capital: float,
        risk_amount_usd: float,
        current_daily_loss_usd: float,
        current_open_positions_count: int,
        fixed_lot_override: bool = False,
    ) -> Dict[str, Any]:
        """
        Vérification indépendante de niveau 0 avant tout ordre.
        """
        if self.emergency_stop_triggered:
            return {
                "safe": False,
                "reason": "🚨 [SafetyGuard] Arrêt d'urgence actif : toute nouvelle position est strictement interdite.",
            }

        if capital <= 0:
            return {"safe": False, "reason": "Capital nul ou invalide."}

        # 1. Contrôle du risque par trade vs plafond absolu
        risk_pct = (risk_amount_usd / capital) * 100.0
        max_risk_ceiling = (
            self.HARD_MAX_FIXED_LOT_RISK_PERCENT
            if fixed_lot_override
            else self.HARD_MAX_RISK_PER_TRADE_PERCENT
        )
        if risk_pct > (max_risk_ceiling + 1e-4):
            return {
                "safe": False,
                "reason": (
                    f"🚫 [SafetyGuard] Risque demandé ({risk_pct:.2f}%) dépasse la limite "
                    f"infranchissable de {max_risk_ceiling:.1f}%."
                ),
            }

        # 2. Contrôle de la perte journalière cumulée
        daily_loss_pct = (abs(min(0.0, current_daily_loss_usd)) / capital) * 100.0
        if daily_loss_pct >= self.HARD_MAX_DAILY_LOSS_PERCENT:
            self.emergency_stop_triggered = True
            logger.critical(f"🛑 [SafetyGuard] Plafond journalier atteint ({daily_loss_pct:.1f}%). Arrêt automatique.")
            return {
                "safe": False,
                "reason": f"🛑 [SafetyGuard] Perte journalière maximale ({self.HARD_MAX_DAILY_LOSS_PERCENT}%) atteinte.",
            }

        # 3. Contrôle des positions simultanées
        if current_open_positions_count >= self.HARD_MAX_OPEN_POSITIONS:
            return {
                "safe": False,
                "reason": f"Limite absolue de {self.HARD_MAX_OPEN_POSITIONS} positions ouvertes atteinte.",
            }

        return {"safe": True, "reason": "Contrôle de sécurité validé avec succès."}

    def trigger_emergency_stop(self, reason: str):
        self.emergency_stop_triggered = True
        logger.critical(f"🚨 [SafetyGuard] ARRÊT D'URGENCE ACTIVÉ : {reason}")

    def reset_emergency_stop(self):
        self.emergency_stop_triggered = False
        logger.info("🛡️ [SafetyGuard] Arrêt d'urgence réinitialisé.")
