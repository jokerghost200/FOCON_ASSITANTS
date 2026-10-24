"""
FATI Knowledge Base (Base de Connaissances Validées)
===================================================
Enregistre les règles et vérités statistiques formellement démontrées.
Empêche d'enregistrer de simples coïncidences comme des vérités établies.
Chaque règle porte son échantillon de validation, sa date, son niveau de confiance et son statut.
"""

from typing import Dict, Any, List, Optional
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    Base de connaissances institutionnelle pour FATI.
    Stocke les règles de trading certifiées et leur niveau de confiance empirique.
    """

    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            self.storage_path = os.path.join(data_dir, "fati_knowledge.json")
        else:
            self.storage_path = storage_path

        self._rules: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self._rules = json.load(f)
            except Exception as e:
                logger.error(f"[KnowledgeBase] Erreur chargement: {e}")
                self._rules = {}

    def _save(self):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self._rules, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[KnowledgeBase] Erreur sauvegarde: {e}")

    def add_rule(
        self,
        rule_id: str,
        statement: str,
        strategy: str,
        evidence: Dict[str, Any],
        conditions: Dict[str, Any],
        status: str = "CERTIFIED",
        confidence: float = 0.85,
    ) -> bool:
        """
        Ajoute ou met à jour une règle certifiée.
        Refuse l'ajout si la taille de l'échantillon empirique est insuffisante (< 20 trades).
        """
        sample_size = int(evidence.get("sample_size", 0))
        if sample_size < 20 and status == "CERTIFIED":
            logger.warning(
                f"[KnowledgeBase] Refus de certification pour '{rule_id}': "
                f"échantillon trop faible ({sample_size} < 20 trades requis)."
            )
            status = "HYPOTHESIS"

        self._rules[rule_id] = {
            "rule_id": rule_id,
            "statement": statement,
            "strategy": strategy,
            "evidence": evidence,
            "conditions": conditions,
            "status": status,
            "confidence": confidence,
            "updated_at": datetime.utcnow().isoformat(),
        }
        self._save()
        logger.info(f"📚 [KnowledgeBase] Règle enregistrée [{status}] : {rule_id} - {statement}")
        return True

    def get_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        return self._rules.get(rule_id)

    def get_certified_rules(self, strategy: Optional[str] = None) -> List[Dict[str, Any]]:
        """Renvoie uniquement les règles validées pour la production."""
        res = [r for r in self._rules.values() if r.get("status") == "CERTIFIED"]
        if strategy:
            res = [r for r in res if r.get("strategy") == strategy]
        return res
