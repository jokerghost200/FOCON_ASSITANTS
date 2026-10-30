# -*- coding: utf-8 -*-
"""
CEREBRO 5 : EL CENTINELA (Pilier 5 - SÉCURISER & PROTÉGER)
==========================================================
Nom Espagnol : El Centinela ("La Sentinelle / Le Gardien")
Rôle : Garde-fou absolu et inviolable assurant la survie du compte.
Supervise le respect des plafonds de risque, contrôle les déploiements de modèles
et déclenche le coupe-circuit ou le rollback instantané en cas de danger.

Composants fédérés :
- SafetyRiskGuard : Limites infranchissables (risque max/trade, perte journalière, stop d'urgence)
- PromotionGate : Porte d'accès sécurisée autorisant uniquement les variantes testées et éprouvées
- RollbackManager : Retour arrière automatique immédiat vers la dernière version saine
- RiskManager (facultatif / délégué) : Calcul précis de lot et protection du capital
"""

import logging
from typing import Dict, Any, Optional

from src.safety.risk_guard import SafetyRiskGuard
from src.safety.promotion_gate import PromotionGate
from src.safety.rollback import RollbackManager
from src.research.model_registry import ModelRegistry
from src.learning.adaptive_weights import AdaptiveWeightsEngine

logger = logging.getLogger(__name__)


class ElCentinela:
    """
    Cerveau Sentinelle & Protection (El Centinela)
    Protection inviolable : ses règles de gestion des risques ne peuvent
    en aucun cas être contournées ni affaiblies par l'apprentissage ou l'IA.
    """

    def __init__(
        self,
        risk_guard: Optional[SafetyRiskGuard] = None,
        promotion_gate: Optional[PromotionGate] = None,
        rollback: Optional[RollbackManager] = None,
        registry: Optional[ModelRegistry] = None,
        weights_engine: Optional[AdaptiveWeightsEngine] = None,
    ):
        self.risk_guard = risk_guard or SafetyRiskGuard()
        self.promotion_gate = promotion_gate or PromotionGate()
        self.rollback = rollback or RollbackManager(
            registry=registry or ModelRegistry(),
            weights_engine=weights_engine,
        )

    def evaluar_seguridad(
        self,
        capital: float,
        risk_amount_usd: float,
        current_daily_loss_usd: float,
        current_open_positions_count: int,
        fixed_lot_override: bool = False,
    ) -> Dict[str, Any]:
        """
        Vérification absolue de sécurité niveau 0 avant l'ouverture de toute position.
        """
        return self.risk_guard.validate_proposal_safety(
            capital=capital,
            risk_amount_usd=risk_amount_usd,
            current_daily_loss_usd=current_daily_loss_usd,
            current_open_positions_count=current_open_positions_count,
            fixed_lot_override=fixed_lot_override,
        )

    def verificar_promocion(
        self,
        candidate_version: str,
        experiment_result: Dict[str, Any],
        walk_forward_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Évalue si une variante de modèle satisfait tous les critères stricts
        d'espérance, de drawdown et de ratio WFE pour être déployée en réel.
        """
        return self.promotion_gate.evaluate_promotion(
            candidate_version=candidate_version,
            experiment_result=experiment_result,
            walk_forward_result=walk_forward_result,
        )

    def activar_rollback(self, razon_incidente: str) -> Dict[str, Any]:
        """
        Déclenche immédiatement le retour arrière vers la configuration stable certifiée.
        """
        logger.critical(f"🛡️ [ElCentinela] Activation du Rollback de sécurité. Motif : {razon_incidente}")
        return self.rollback.trigger_rollback(incident_reason=razon_incidente)

    def activar_parada_emergencia(self, motivo: str = "Arrêt d'urgence Sentinelle"):
        """Active l'arrêt d'urgence absolu (Kill Switch)."""
        logger.critical(f"🛑 [ElCentinela] KILL SWITCH ACTIVÉ : {motivo}")
        self.risk_guard.trigger_emergency_stop(reason=motivo)

    def desactivar_parada_emergencia(self):
        """Réinitialise l'arrêt d'urgence après inspection humaine."""
        logger.warning("🟢 [ElCentinela] Arrêt d'urgence réinitialisé.")
        self.risk_guard.reset_emergency_stop()

    @property
    def parada_emergencia_activa(self) -> bool:
        """Indique si l'arrêt d'urgence est actuellement engagé."""
        return self.risk_guard.emergency_stop_triggered
