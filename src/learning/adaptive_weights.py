"""
Moteur de pondération adaptative (Adaptive Weights Engine).
Ajuste dynamiquement les coefficients d'importance des confluences (Order Block, Boomerang, Trend, RSI)
selon le taux de réussite empirique réel constaté sur les derniers trades.
"""
import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "ob_weight": 35.0,
    "boomerang_weight": 30.0,
    "trend_ema_weight": 15.0,
    "rsi_pullback_weight": 15.0,
    "htf_alignment_weight": 15.0,
    "amd_expansion_weight": 10.0,
}

WEIGHT_BOUNDS = {
    "ob_weight": (15.0, 55.0),
    "boomerang_weight": (10.0, 50.0),
    "trend_ema_weight": (5.0, 30.0),
    "rsi_pullback_weight": (5.0, 30.0),
    "htf_alignment_weight": (5.0, 30.0),
    "amd_expansion_weight": (0.0, 25.0),
}


class AdaptiveWeightsEngine:
    """
    Gestionnaire des poids dynamiques de scoring institutionnel.
    Fait évoluer mathématiquement les règles sans risquer de corrompre le code source.
    """
    def __init__(self, file_path: str = "data/adaptive_weights.json", learning_rate: float = 0.15):
        self.file_path = file_path
        self.learning_rate = learning_rate
        self.weights = dict(DEFAULT_WEIGHTS)
        self.history_log = []
        self._load()

    def _load(self):
        """Charge les poids adaptatifs depuis le disque s'ils existent."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    saved_w = data.get("weights", {})
                    for k, v in saved_w.items():
                        if k in DEFAULT_WEIGHTS:
                            low, high = WEIGHT_BOUNDS.get(k, (0.0, 100.0))
                            self.weights[k] = max(low, min(high, float(v)))
                    self.history_log = data.get("history_log", [])[-20:]
                logger.info(f"🧠 [Poids Adaptatifs] Poids chargés avec succès: {self.weights}")
            except Exception as e:
                logger.warning(f"Erreur chargement adaptive_weights.json ({e}). Utilisation des poids par défaut.")
                self.weights = dict(DEFAULT_WEIGHTS)
        else:
            self._save()

    def _save(self):
        """Sauvegarde les poids calibrés de manière atomique."""
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            temp_path = self.file_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump({
                    "weights": self.weights,
                    "default_weights": DEFAULT_WEIGHTS,
                    "bounds": WEIGHT_BOUNDS,
                    "history_log": self.history_log,
                }, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self.file_path)
        except Exception as e:
            logger.error(f"Erreur sauvegarde adaptive_weights.json : {e}")

    def get_weights(self) -> Dict[str, float]:
        """Retourne une copie des poids actifs actuels."""
        return dict(self.weights)

    def record_trade_outcome(self, setup_type: str, is_win: bool, pnl: float = 0.0):
        """Ajuste directement et prudemment le poids associé à un setup après clôture."""
        key_map = {
            "ORDER_BLOCK": "ob_weight",
            "SMC_ORDER_BLOCK": "ob_weight",
            "BOOMERANG": "boomerang_weight",
            "TREND": "trend_ema_weight",
            "TREND_EMA": "trend_ema_weight",
            "RSI": "rsi_pullback_weight",
            "AMD": "amd_expansion_weight",
            "AMD_EXPANSION": "amd_expansion_weight",
        }
        target = key_map.get(setup_type.upper(), "ob_weight")
        delta = self.learning_rate if is_win else -self.learning_rate
        low, high = WEIGHT_BOUNDS.get(target, (0.0, 100.0))
        self.weights[target] = max(low, min(high, round(self.weights.get(target, 30.0) + delta, 2)))
        self._save()

    def recalibrate(self, closed_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Recalibre les poids en analysant la performance empirique des composantes.
        Nécessite au moins 3 trades clôturés pour commencer l'ajustement.
        """
        if not closed_trades or len(closed_trades) < 3:
            return {
                "status": "SKIPPED",
                "reason": f"Pas assez de trades ({len(closed_trades)} < 3 minimum requis pour recalibrer).",
                "weights": dict(self.weights),
            }

        # Fenêtre d'analyse : jusqu'à 30 derniers trades
        sample = closed_trades[-30:]
        
        # Statistiques par type de setup
        stats = {
            "ob": {"wins": 0, "losses": 0, "total": 0},
            "boomerang": {"wins": 0, "losses": 0, "total": 0},
            "trend": {"wins": 0, "losses": 0, "total": 0},
            "trend_ema": {"wins": 0, "losses": 0, "total": 0},
            "rsi": {"wins": 0, "losses": 0, "total": 0},
            "amd_expansion": {"wins": 0, "losses": 0, "total": 0},
        }

        for t in sample:
            profit = float(t.get("final_profit_usd", 0.0) or 0.0)
            is_win = profit > 0.05
            is_loss = profit < -0.05
            snapshot = t.get("snapshot", {}) or {}

            # OB
            if snapshot.get("has_ob"):
                stats["ob"]["total"] += 1
                if is_win: stats["ob"]["wins"] += 1
                elif is_loss: stats["ob"]["losses"] += 1

            # Boomerang
            if snapshot.get("has_boomerang"):
                stats["boomerang"]["total"] += 1
                if is_win: stats["boomerang"]["wins"] += 1
                elif is_loss: stats["boomerang"]["losses"] += 1

            # Trend HTF
            if snapshot.get("htf_trend") and "NEUTRAL" not in str(snapshot.get("htf_trend")):
                stats["trend"]["total"] += 1
                if is_win: stats["trend"]["wins"] += 1
                elif is_loss: stats["trend"]["losses"] += 1

            # Trend EMA
            if abs(float(snapshot.get("ema_dist_atr", 0.0) or 0.0)) > 0.0:
                stats["trend_ema"]["total"] += 1
                if is_win: stats["trend_ema"]["wins"] += 1
                elif is_loss: stats["trend_ema"]["losses"] += 1

            # RSI Pullback
            rsi = float(snapshot.get("rsi_14", 50.0) or 50.0)
            if 40.0 <= rsi <= 60.0:
                stats["rsi"]["total"] += 1
                if is_win: stats["rsi"]["wins"] += 1
                elif is_loss: stats["rsi"]["losses"] += 1

            # AMD SMC Expansion
            amd_p = str(snapshot.get("amd_phase", "")).upper()
            fvg_v = str(snapshot.get("fvg", "None"))
            if ("EXPANSION" in amd_p or "DISTRIBUTION" in amd_p or "MANIPULATION" in amd_p) and fvg_v != "None":
                stats["amd_expansion"]["total"] += 1
                if is_win: stats["amd_expansion"]["wins"] += 1
                elif is_loss: stats["amd_expansion"]["losses"] += 1

        adjustments = {}
        lr = self.learning_rate

        # ── Ajustement Order Block ──
        if stats["ob"]["total"] >= 2:
            wr = stats["ob"]["wins"] / stats["ob"]["total"]
            target = DEFAULT_WEIGHTS["ob_weight"] * (0.6 + 0.8 * wr) # Ex: 50% WR -> x1.0, 80% WR -> x1.24
            target = max(WEIGHT_BOUNDS["ob_weight"][0], min(WEIGHT_BOUNDS["ob_weight"][1], target))
            old_w = self.weights["ob_weight"]
            self.weights["ob_weight"] = round(old_w * (1 - lr) + target * lr, 1)
            adjustments["ob_weight"] = {"old": old_w, "new": self.weights["ob_weight"], "winrate": round(wr * 100, 1)}

        # ── Ajustement Boomerang ──
        if stats["boomerang"]["total"] >= 2:
            wr = stats["boomerang"]["wins"] / stats["boomerang"]["total"]
            target = DEFAULT_WEIGHTS["boomerang_weight"] * (0.6 + 0.8 * wr)
            target = max(WEIGHT_BOUNDS["boomerang_weight"][0], min(WEIGHT_BOUNDS["boomerang_weight"][1], target))
            old_w = self.weights["boomerang_weight"]
            self.weights["boomerang_weight"] = round(old_w * (1 - lr) + target * lr, 1)
            adjustments["boomerang_weight"] = {"old": old_w, "new": self.weights["boomerang_weight"], "winrate": round(wr * 100, 1)}

        # ── Ajustement Trend HTF ──
        if stats["trend"]["total"] >= 2:
            wr = stats["trend"]["wins"] / stats["trend"]["total"]
            target = DEFAULT_WEIGHTS["htf_alignment_weight"] * (0.6 + 0.8 * wr)
            target = max(WEIGHT_BOUNDS["htf_alignment_weight"][0], min(WEIGHT_BOUNDS["htf_alignment_weight"][1], target))
            old_w = self.weights["htf_alignment_weight"]
            self.weights["htf_alignment_weight"] = round(old_w * (1 - lr) + target * lr, 1)
            adjustments["htf_alignment_weight"] = {"old": old_w, "new": self.weights["htf_alignment_weight"], "winrate": round(wr * 100, 1)}

        # ── Ajustement Trend EMA ──
        if stats["trend_ema"]["total"] >= 2:
            wr = stats["trend_ema"]["wins"] / stats["trend_ema"]["total"]
            target = DEFAULT_WEIGHTS["trend_ema_weight"] * (0.6 + 0.8 * wr)
            target = max(WEIGHT_BOUNDS["trend_ema_weight"][0], min(WEIGHT_BOUNDS["trend_ema_weight"][1], target))
            old_w = self.weights["trend_ema_weight"]
            self.weights["trend_ema_weight"] = round(old_w * (1 - lr) + target * lr, 1)
            adjustments["trend_ema_weight"] = {"old": old_w, "new": self.weights["trend_ema_weight"], "winrate": round(wr * 100, 1)}

        # ── Ajustement RSI Pullback ──
        if stats["rsi"]["total"] >= 2:
            wr = stats["rsi"]["wins"] / stats["rsi"]["total"]
            target = DEFAULT_WEIGHTS["rsi_pullback_weight"] * (0.6 + 0.8 * wr)
            target = max(WEIGHT_BOUNDS["rsi_pullback_weight"][0], min(WEIGHT_BOUNDS["rsi_pullback_weight"][1], target))
            old_w = self.weights["rsi_pullback_weight"]
            self.weights["rsi_pullback_weight"] = round(old_w * (1 - lr) + target * lr, 1)
            adjustments["rsi_pullback_weight"] = {"old": old_w, "new": self.weights["rsi_pullback_weight"], "winrate": round(wr * 100, 1)}

        # ── Ajustement AMD Expansion ──
        if stats["amd_expansion"]["total"] >= 2:
            wr = stats["amd_expansion"]["wins"] / stats["amd_expansion"]["total"]
            target = DEFAULT_WEIGHTS["amd_expansion_weight"] * (0.6 + 0.8 * wr)
            target = max(WEIGHT_BOUNDS["amd_expansion_weight"][0], min(WEIGHT_BOUNDS["amd_expansion_weight"][1], target))
            old_w = self.weights["amd_expansion_weight"]
            self.weights["amd_expansion_weight"] = round(old_w * (1 - lr) + target * lr, 1)
            adjustments["amd_expansion_weight"] = {"old": old_w, "new": self.weights["amd_expansion_weight"], "winrate": round(wr * 100, 1)}

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "sample_size": len(sample),
            "adjustments": adjustments,
            "current_weights": dict(self.weights),
        }
        self.history_log.append(log_entry)
        self.history_log = self.history_log[-20:]
        self._save()

        logger.info(f"🧠 [Poids Adaptatifs] Recalibrage terminé ({len(sample)} trades analysés) : {adjustments}")
        return {
            "status": "RECALIBRATED",
            "sample_size": len(sample),
            "adjustments": adjustments,
            "weights": dict(self.weights),
        }

    def reset_to_defaults(self):
        """Réinitialise les poids aux valeurs de base sécurisées."""
        self.weights = dict(DEFAULT_WEIGHTS)
        self._save()
        logger.info("🧠 [Poids Adaptatifs] Réinitialisation aux valeurs initiales d'usine effectuée.")
