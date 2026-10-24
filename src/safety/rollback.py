"""
FATI Rollback Manager (Gestionnaire de Retour Arrière et Restauration Automatique)
=================================================================================
Permet de restaurer instantanément la dernière configuration ou version de modèle stable :
- Déclenchement automatique en cas de dérive de performance critique en réel
- Rétablissement des pondérations conservatrices et archivage de la version défaillante
- Traçabilité complète de l'incident
"""

from typing import Dict, Any, Optional
import logging
from src.research.model_registry import ModelRegistry
from src.learning.adaptive_weights import AdaptiveWeightsEngine

logger = logging.getLogger(__name__)


class RollbackManager:
    """
    Gestionnaire de rollback autonome.
    Protège le capital en restaurant une version antérieure validée en cas d'anomalie.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        weights_engine: Optional[AdaptiveWeightsEngine] = None,
        fallback_stable_version: str = "v1.0.0",
    ):
        self.registry = registry or ModelRegistry()
        self.weights_engine = weights_engine
        self.fallback_stable_version = fallback_stable_version

    def trigger_rollback(self, incident_reason: str) -> Dict[str, Any]:
        """
        Exécute le rollback immédiat vers la version de secours certifiée.
        """
        active_model = self.registry.get_active_model() or {}
        current_ver = active_model.get("version", "UNKNOWN")

        logger.critical(
            f"🚨 [RollbackManager] DÉCLENCHEMENT DU ROLLBACK pour {current_ver} -> {self.fallback_stable_version}. "
            f"Raison : {incident_reason}"
        )

        # 1. Rétablir le modèle de repli dans le registre
        success = self.registry.activate_model(self.fallback_stable_version)

        # 2. Réinitialiser les poids adaptatifs aux valeurs saines de base
        if self.weights_engine:
            self.weights_engine.weights = {
                "ob_weight": 35.0,
                "boomerang_weight": 30.0,
                "trend_ema_weight": 15.0,
                "rsi_pullback_weight": 15.0,
                "htf_alignment_weight": 15.0,
                "amd_expansion_weight": 10.0,
            }
            self.weights_engine._save_weights()
            logger.info("🛡️ [RollbackManager] Poids adaptatifs réinitialisés aux valeurs équilibrées de base.")

        return {
            "success": success,
            "previous_version": current_ver,
            "restored_version": self.fallback_stable_version,
            "incident_reason": incident_reason,
        }
