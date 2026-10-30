"""
FATI Experience Memory (Mémoire Structurée des Expériences)
===========================================================
Conserve et structure les expériences de trading sous forme de cas d'apprentissage :
- Regroupement par famille de stratégie (CRT, SMC, Boomerang, Trend)
- Détection des clusters d'échecs (ex: fakeouts en session asiatique, contre-tendance HTF)
- Calcul du taux de succès contextuel par symbole et régime
"""

from typing import Dict, Any, List, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ExperienceMemory:
    """
    Mémoire de niveau 2 (Expériences analysées).
    Sépare les faits bruts de la signification statistique.
    """

    def __init__(self):
        self._experiences: List[Dict[str, Any]] = []
        self._error_clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._success_clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add_experience(self, trade_record: Dict[str, Any]):
        """Ajoute et indexe une expérience de trading terminée."""
        if not trade_record:
            return

        # Protection anti-doublon par ticket MT5
        tkt = trade_record.get("ticket")
        if tkt is not None and any(e.get("ticket") == tkt for e in self._experiences):
            return

        result_r = float(trade_record.get("result_r") or 0.0)
        strategy = str(trade_record.get("strategy") or "GENERIC")
        regime = str(trade_record.get("market_regime") or "UNKNOWN")
        session = str(trade_record.get("session") or "UNKNOWN")
        symbol = str(trade_record.get("symbol") or "UNKNOWN")

        exp = {
            "ticket": trade_record.get("ticket"),
            "symbol": symbol,
            "strategy": strategy,
            "regime": regime,
            "session": session,
            "result_r": result_r,
            "exit_reason": trade_record.get("exit_reason", ""),
            "confluences": trade_record.get("confluences", []),
            "metadata": trade_record.get("metadata", {}),
        }
        self._experiences.append(exp)

        # Indexation par clé contextuelle
        context_key = f"{strategy}::{regime}::{session}"
        if result_r < 0:
            self._error_clusters[context_key].append(exp)
        else:
            self._success_clusters[context_key].append(exp)

    def get_cluster_stats(self, strategy: str, regime: str, session: str) -> Dict[str, Any]:
        """Retourne les statistiques empiriques observées pour un contexte donné."""
        key = f"{strategy}::{regime}::{session}"
        losses = self._error_clusters.get(key, [])
        wins = self._success_clusters.get(key, [])
        total = len(losses) + len(wins)
        if total == 0:
            return {"sample_size": 0, "win_rate": 0.0, "expectancy_r": 0.0, "is_statistically_valid": False}

        win_rate = len(wins) / total
        total_r = sum(e["result_r"] for e in wins) + sum(e["result_r"] for e in losses)
        exp_r = total_r / total

        return {
            "sample_size": total,
            "win_rate": round(win_rate, 3),
            "expectancy_r": round(exp_r, 3),
            "wins": len(wins),
            "losses": len(losses),
            "is_statistically_valid": total >= 15,  # Seuil minimal pour inférence robuste
        }

    def detect_toxic_patterns(self, min_sample: int = 5, max_win_rate: float = 0.30) -> List[Dict[str, Any]]:
        """
        Détecte les contextes récurrents destructeurs de capital
        (ex: une stratégie qui perd systématiquement dans un régime donné).
        """
        toxic = []
        all_keys = set(self._error_clusters.keys()).union(set(self._success_clusters.keys()))
        for key in all_keys:
            losses = len(self._error_clusters.get(key, []))
            wins = len(self._success_clusters.get(key, []))
            total = losses + wins
            if total >= min_sample:
                wr = wins / total
                if wr <= max_win_rate:
                    parts = key.split("::")
                    toxic.append({
                        "context_key": key,
                        "strategy": parts[0] if len(parts) > 0 else "UNKNOWN",
                        "regime": parts[1] if len(parts) > 1 else "UNKNOWN",
                        "session": parts[2] if len(parts) > 2 else "UNKNOWN",
                        "sample_size": total,
                        "win_rate": round(wr, 3),
                        "recommendation": f"Suspendre ou filtrer {parts[0]} en régime {parts[1]} ({parts[2]}).",
                    })
        return toxic
