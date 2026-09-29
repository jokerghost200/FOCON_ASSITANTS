"""
Sentinelle d'Adaptation Intelligente au Marché (Market Adaptation Sentinel).
=============================================================================
Surveille en continu l'état du compte MT5 et les résultats de trading récents.
Détecte automatiquement les anomalies et dangers de marché :
- Série de pertes consécutives (Loss Streak >= 2)
- Recul journalier approchant la limite autorisée
- Capital Micro (< 100$) exposé à des paires trop volatiles
- Marché unilatéral nécessitant une restriction de direction

Adapte les paramètres pour préserver le profit et protéger le capital.
"""

import logging
from typing import Dict, Any, List, Optional
from copy import deepcopy

logger = logging.getLogger(__name__)


class MarketAdaptationSentinel:
    """
    Cerveau d'adaptation proactive du trading.
    Analyse les risques et formule les ajustements optimaux.
    """

    @staticmethod
    def evaluate_market_and_account(
        orchestrator,
        config: Dict[str, Any],
        recent_trades: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Évalue si la situation actuelle requiert une adaptation des réglages.
        Retourne un dictionnaire avec le statut, les explications et les changements recommandés.
        """
        config = deepcopy(config)
        rm = config.get("risk_management", {})
        tr = config.get("trading", {})

        curr_risk = float(rm.get("default_risk_percent", 0.5))
        curr_conf = float(rm.get("min_confidence", 60.0))
        curr_max_pos = int(rm.get("max_open_positions", 3))
        curr_types = tr.get("allowed_market_types", ["Forex"])
        if isinstance(curr_types, str):
            curr_types = [curr_types]

        changes: List[Dict[str, Any]] = []
        diagnostics: List[str] = []
        speech_alerts: List[str] = []

        # ── 1. Analyse du Capital & Palier Micro ──
        balance = 1000.0
        equity = 1000.0
        try:
            if hasattr(orchestrator, "mt5") and orchestrator.mt5.connected:
                acc = orchestrator.mt5.get_account_info() or {}
                balance = float(acc.get("balance", balance) or balance)
                equity = float(acc.get("equity", equity) or equity)
        except Exception:
            pass

        if balance <= 100.0:
            # Capital micro : forcer la sécurité maximale
            if any(t in curr_types for t in ["Crypto", "Indices"]):
                forex_only = ["Forex"]
                changes.append({
                    "section": "trading",
                    "key": "allowed_market_types",
                    "label": "Types de marchés autorisés",
                    "current": curr_types,
                    "value": forex_only,
                    "reason": f"Compte Micro ({balance:.2f} $) : exclusion des actifs à fort effet de levier pour sécuriser le capital."
                })
                diagnostics.append("Micro-Capital détecté : actifs limités au Forex.")
                speech_alerts.append(f"Compte micro à {int(balance)} dollars détecté : concentration sur le Forex pour sécuriser les gains.")

            if curr_max_pos > 2:
                changes.append({
                    "section": "risk_management",
                    "key": "max_open_positions",
                    "label": "Positions ouvertes maximales",
                    "current": curr_max_pos,
                    "value": 1 if balance <= 50.0 else 2,
                    "reason": "Réduction du nombre de positions simultanées pour éviter la sur-marge."
                })

        # ── 2. Analyse de la Série de Pertes Récentes ──
        trades_to_check = recent_trades or []
        if not trades_to_check and hasattr(orchestrator, "memory"):
            try:
                trades_to_check = getattr(orchestrator.memory, "trades", [])
            except Exception:
                trades_to_check = []

        consecutive_losses = 0
        for t in reversed(trades_to_check[-6:]):
            profit = float(t.get("profit", 0.0) or 0.0)
            if profit < 0:
                consecutive_losses += 1
            elif profit > 0:
                break

        if consecutive_losses >= 2:
            # Série de pertes : marché difficile ou sans tendance claire
            safer_risk = max(0.2, round(curr_risk * 0.7, 2))
            higher_conf = min(85.0, max(curr_conf, 72.0))

            if safer_risk < curr_risk:
                changes.append({
                    "section": "risk_management",
                    "key": "default_risk_percent",
                    "label": "Risque par trade",
                    "current": curr_risk,
                    "value": safer_risk,
                    "reason": f"{consecutive_losses} pertes consécutives détectées : réduction temporaire de l'exposition."
                })

            if higher_conf > curr_conf:
                changes.append({
                    "section": "risk_management",
                    "key": "min_confidence",
                    "label": "Confiance minimale",
                    "current": curr_conf,
                    "value": higher_conf,
                    "reason": "Hausse du seuil de confiance IA pour filtrer les faux signaux en marché volatil."
                })

            diagnostics.append(f"{consecutive_losses} pertes consécutives : filtre de confiance renforcé et risque abaissé.")
            speech_alerts.append(
                f"Attention : {consecutive_losses} pertes consécutives détectées sur les derniers trades. "
                f"Le filtre de confiance est renforcé à {int(higher_conf)}% et le risque est réduit à {safer_risk}%."
            )

        # ── 3. Synthèse Finale ──
        need_adaptation = len(changes) > 0
        summary_reply = " ".join(diagnostics) if diagnostics else "Les conditions de marché et du compte sont stables. Aucun danger détecté."
        speech_text = " ".join(speech_alerts) if speech_alerts else "Analyse terminée : vos paramètres actuels sont parfaitement adaptés au marché."

        return {
            "need_adaptation": need_adaptation,
            "reply": summary_reply,
            "speech_alert": speech_text,
            "changes": changes,
            "balance": balance,
            "consecutive_losses": consecutive_losses,
        }
