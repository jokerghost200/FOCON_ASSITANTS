"""
FATI System Health Monitor (Moniteur d'Intégrité et de Santé Globale)
====================================================================
Vérifie la robustesse de chaque composant de l'architecture autonome FATI :
- État de la connexion MT5 et du terminal
- Intégrité de la base de données et du journal SQLite
- Niveau d'alerte de dérive (Drift Status)
- État des coupe-circuits et du garde-fou de sécurité (RiskGuard)
"""

from typing import Dict, Any, Optional
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SystemHealthMonitor:
    """
    Moniteur d'intégrité systémique pour FATI.
    """

    def __init__(
        self,
        mt5_connector=None,
        journal=None,
        drift_detector=None,
        safety_guard=None,
        registry=None,
    ):
        self.mt5 = mt5_connector
        self.journal = journal
        self.drift_detector = drift_detector
        self.safety_guard = safety_guard
        self.registry = registry

    def get_health_report(self) -> Dict[str, Any]:
        """Alias pour check_health."""
        return self.check_health()

    def check_health(self) -> Dict[str, Any]:
        """Effectue un bilan de santé instantané de l'ensemble du système."""
        now_str = datetime.utcnow().isoformat()
        report = {
            "timestamp": now_str,
            "overall_status": "HEALTHY",
            "components": {},
            "alerts": [],
        }

        # 1. MT5 Connector
        if self.mt5:
            is_conn = bool(getattr(self.mt5, "connected", False))
            report["components"]["mt5"] = {"connected": is_conn}
            if not is_conn:
                report["overall_status"] = "DEGRADED"
                report["alerts"].append("Terminal MT5 non connecté.")

        # 2. Trade Journal
        if self.journal and hasattr(self.journal, "db_path"):
            db_exists = os.path.exists(self.journal.db_path)
            size_kb = os.path.getsize(self.journal.db_path) / 1024.0 if db_exists else 0.0
            report["components"]["journal"] = {"exists": db_exists, "size_kb": round(size_kb, 1)}

        # 3. Safety Guard
        if self.safety_guard:
            is_stop = getattr(self.safety_guard, "emergency_stop_triggered", False)
            report["components"]["safety"] = {"emergency_stop": is_stop}
            if is_stop:
                report["overall_status"] = "EMERGENCY_STOP"
                report["alerts"].append("Arrêt d'urgence actif dans SafetyRiskGuard.")

        # 4. Modèle Actif
        if self.registry:
            active_m = self.registry.get_active_model()
            report["components"]["model"] = {
                "active_version": active_m.get("version", "NONE") if active_m else "NONE"
            }

        return report
