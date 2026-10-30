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
        open_positions_risk_usd: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Vérification absolue de sécurité niveau 0 avant l'ouverture de toute position.
        Inclut également le plafond de risque cumulé du portefeuille (open_positions_risk_usd + risk_amount_usd).
        """
        res = self.risk_guard.validate_proposal_safety(
            capital=capital,
            risk_amount_usd=risk_amount_usd,
            current_daily_loss_usd=current_daily_loss_usd,
            current_open_positions_count=current_open_positions_count,
            fixed_lot_override=fixed_lot_override,
        )
        if not res.get("safe", False):
            return res

        # Contrôle du risque cumulé simultané sur toutes les positions ouvertes (plafond 12% du capital)
        if capital > 0 and open_positions_risk_usd > 0 and not fixed_lot_override:
            total_portfolio_risk = float(open_positions_risk_usd) + float(max(0.0, risk_amount_usd))
            max_portfolio_risk_pct = 0.12  # 12% max d'exposition simultanée totale
            if (total_portfolio_risk / capital) > max_portfolio_risk_pct:
                return {
                    "safe": False,
                    "reason": (
                        f"Plafond de risque cumulé portefeuille dépassé : "
                        f"${total_portfolio_risk:.2f} / ${capital:.2f} "
                        f"({(total_portfolio_risk / capital) * 100:.1f}% > {max_portfolio_risk_pct * 100:.0f}%)."
                    ),
                }
        return res

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

    @staticmethod
    def _extract_currency_exposures(symbol: str, action: str) -> Dict[str, int]:
        """
        Décompose un symbole (ex: EURUSD, XAUUSD, GBPJPY) en exposition directionnelle par devise.
        BUY EURUSD -> {'EUR': +1, 'USD': -1}
        SELL EURUSD -> {'EUR': -1, 'USD': +1}
        """
        sym = "".join(ch for ch in str(symbol or "").upper() if ch.isalpha())
        act = str(action or "").upper()
        if act not in ("BUY", "SELL") or len(sym) < 6:
            return {}
        sign = 1 if act == "BUY" else -1
        base = sym[:3]
        quote = sym[3:6]
        known = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "XAU", "XAG", "BTC", "ETH"}
        exp = {}
        if base in known:
            exp[base] = sign
        if quote in known:
            exp[quote] = -sign
        return exp

    def verificar_correlacion_divisas(
        self,
        symbol: str,
        action: str,
        open_positions: list,
        max_same_currency_exposure: int = 1,
    ) -> Dict[str, Any]:
        """
        Filtre Anti-Corrélation des Devises :
        Empêche d'ouvrir plusieurs positions exposées dans le même sens sur la même devise
        (ex: BUY EURUSD + BUY GBPUSD + BUY XAUUSD = triple vente USD simultanée).
        """
        if not open_positions:
            return {"safe": True, "reason": "Aucune position ouverte."}

        target_exp = self._extract_currency_exposures(symbol, action)
        if not target_exp:
            return {"safe": True, "reason": "Symbole non soumis au filtre de corrélation."}

        max_allowed = max(1, int(max_same_currency_exposure or 1))
        for ccy, dir_sign in target_exp.items():
            same_dir_symbols = []
            for pos in open_positions:
                if not isinstance(pos, dict):
                    continue
                p_sym = str(pos.get("symbol", ""))
                p_type = str(pos.get("type", "")).upper()
                if p_sym.upper() == str(symbol).upper():
                    continue
                p_exp = self._extract_currency_exposures(p_sym, p_type)
                if p_exp.get(ccy) == dir_sign:
                    same_dir_symbols.append(f"{p_sym} ({p_type})")

            if len(same_dir_symbols) >= max_allowed:
                dir_txt = "HAUSSE (Long)" if dir_sign > 0 else "BAISSE (Short)"
                return {
                    "safe": False,
                    "reason": (
                        f"🛡️ [Anti-Corrélation {ccy}] Exposition {dir_txt} sur {ccy} déjà active via "
                        f"{', '.join(same_dir_symbols)}. Ordre {symbol} ({action}) bloqué pour éviter de doubler le risque."
                    ),
                }

        return {"safe": True, "reason": "Exposition devises équilibrée."}
