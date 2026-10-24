"""
FATI Model Registry (Registre Versionné des Modèles et Configurations de Trading)
=================================================================================
Conserve l'historique et les spécifications de chaque modèle ou configuration testée :
- Versionnage semver (ex: core_v1.0.0, core_v1.1.0)
- Paramètres complets (seuils, filtres, poids)
- Métriques validées hors-échantillon (OOS Expectancy, Drawdown, Sharpe)
- Gestion du cycle de vie : CANDIDATE -> ACTIVE -> ARCHIVED / ROLLED_BACK
"""

from typing import Dict, Any, List, Optional
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registre central de modèles pour FATI.
    Garantit la traçabilité et permet un rollback instantané vers une version stable.
    """

    def __init__(self, registry_file: Optional[str] = None, storage_path: Optional[str] = None):
        target = storage_path or registry_file
        if target is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            self.registry_file = os.path.join(data_dir, "fati_model_registry.json")
        else:
            self.registry_file = target

        self._registry: Dict[str, Dict[str, Any]] = {}
        self._load()

        # Initialiser une version par défaut v1.0.0 si le registre est vide
        if not self._registry:
            self.register_model(
                version="v1.0.0",
                name="FATI Core Baseline",
                parameters={"min_confidence": 65.0, "min_rr": 1.5, "risk_percent": 1.0},
                metrics={"expectancy_r": 0.25, "sharpe_ratio": 1.2, "status": "ACTIVE"},
                status="ACTIVE",
            )

    def _load(self):
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    self._registry = json.load(f)
            except Exception as e:
                logger.error(f"[ModelRegistry] Erreur chargement: {e}")
                self._registry = {}

    def _save(self):
        try:
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(self._registry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[ModelRegistry] Erreur sauvegarde: {e}")

    def register_model(
        self,
        version: str,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        author: str = "Quant",
        description: str = "",
        status: str = "CANDIDATE",
    ) -> bool:
        """Enregistre une nouvelle version de modèle."""
        self._registry[version] = {
            "version": version,
            "name": name,
            "parameters": parameters or {},
            "metrics": metrics or {},
            "author": author,
            "description": description,
            "status": status,
            "registered_at": datetime.utcnow().isoformat(),
        }
        self._save()
        logger.info(f"🏷️ [ModelRegistry] Modèle enregistré : {version} ({name}) [{status}]")
        return True

    def get_active_model(self) -> Optional[Dict[str, Any]]:
        """Renvoie le modèle actuellement actif en production."""
        for m in self._registry.values():
            if m.get("status") == "ACTIVE":
                return m
        return None

    def activate_model(self, version: str) -> bool:
        """Promeut une version au statut ACTIVE et archive l'ancienne."""
        if version not in self._registry:
            logger.error(f"[ModelRegistry] Version {version} inconnue.")
            return False

        for k, v in self._registry.items():
            if v.get("status") == "ACTIVE":
                v["status"] = "ARCHIVED"

        self._registry[version]["status"] = "ACTIVE"
        self._registry[version]["activated_at"] = datetime.utcnow().isoformat()
        self._save()
        logger.info(f"🚀 [ModelRegistry] Modèle {version} activé avec succès en production.")
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        return list(self._registry.values())
