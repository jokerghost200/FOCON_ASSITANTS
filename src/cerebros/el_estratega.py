# -*- coding: utf-8 -*-
"""
CEREBRO 2 : EL ESTRATEGA (Pilier 2 - RÉFLÉCHIR & DÉCIDER)
=========================================================
Nom Espagnol : El Estratega ("Le Stratège")
Rôle : L'Arbitre et Décideur Officiel Unique de FOCON ASSISTANT.
Formule, évalue et sélectionne les meilleures décisions de trading
en combinant les setups institutionnels, la calibration logistique, l'espérance mathématique E(R)
et confronte de façon critique les recommandations de l'IA externe (OpenRouter / Gemini).

Composants fédérés :
- LocalQuantBrain : Cerveau quantitatif local, probabilités logistiques sigmoïdes, espérance E(R)
- DecisionEngine : Arbitrage global des setups, filtres NO-TRADE
- CRTDetector : Candle Range Theory (Sweeps C1, FVG, Pending Limits)
- OrderBlockDetector : Smart Money Concepts (Order Blocks institutionnels)
- BoomerangDetector : Rebond EMA 50/200 & Rejet ZigZag
"""

import logging
from typing import Dict, Any, List, Optional

from src.ai.local_quant_brain import LocalQuantBrain
from src.core.decision_engine import DecisionEngine
from src.core.crt_detector import CRTDetector
from src.core.order_block_detector import OrderBlockDetector
from src.core.boomerang_detector import BoomerangDetector

logger = logging.getLogger(__name__)


class ElEstratega:
    """
    Cerveau Stratège (El Estratega) - DÉCIDEUR OFFICIEL UNIQUE.
    Analyse les confluences, applique les règles de gestion institutionnelles,
    arbitre les avis de l'IA externe et produit des signaux rigoureux à espérance mathématique positive.
    """

    def __init__(
        self,
        local_brain: Optional[LocalQuantBrain] = None,
        decision_engine: Optional[DecisionEngine] = None,
        observador=None,
        weights_engine=None,
        drift_detector=None,
    ):
        self.weights_engine = weights_engine
        self.drift_detector = drift_detector
        self.local_brain = local_brain or LocalQuantBrain(min_confidence=60.0, min_rr=1.5, weights_engine=weights_engine)
        self.decision_engine = decision_engine or DecisionEngine(
            local_brain=self.local_brain,
            regime_detector=getattr(observador, "regime_detector", None),
            drift_detector=drift_detector,
        )
        self.crt_detector = CRTDetector(min_sweep_atr_mult=0.10, min_range_atr_mult=0.50, min_rr_tp1=1.5)
        self.ob_detector = OrderBlockDetector(impulse_atr_mult=1.8, lookback=150, retest_tolerance=0.6)
        self.boomerang = BoomerangDetector(impulse_atr_mult=2.0, min_rr=1.5)

    def decidir(
        self,
        symbol: str,
        timeframe: str,
        tech_features: Dict[str, Any],
        amd_context: Optional[Dict[str, Any]] = None,
        boomerang_data: Optional[Dict[str, Any]] = None,
        order_blocks: Optional[List[Any]] = None,
        crt_setup: Optional[Any] = None,
        htf_trend: str = "Unknown",
        spread_price: float = 0.0,
        direction_filter: str = "BOTH",
        observation: Optional[Dict[str, Any]] = None,
        ai_advice: Optional[Dict[str, Any]] = None,
        economic_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Décideur Officiel Unique :
        Génère une décision stratégique complète par arbitrage institutionnel et quantitatif.
        Si une observation structurée d'El Observador est fournie, extrait automatiquement les contextes.
        Si un contexte macroéconomique d'El Economista est fourni, respecte le gel d'annonces (News Blackout)
        et ajuste la confiance en fonction de la force relative des devises (CSI) et du carry trade.
        Si une alerte de dérive (DriftDetector) est active, applique une mise en quarantaine immédiate (NO-TRADE).
        Si un avis d'IA externe est fourni, le confronte rigoureusement à l'espérance mathématique.
        """
        # 0.0 Évaluation de la dérive de performance (DriftDetector / Quarantaine)
        if self.drift_detector and getattr(self.drift_detector, "is_quarantined", False):
            reason = "Stratégie en QUARANTAINE CRITIQUE (DriftDetector) : dérive de performance négative."
            logger.critical(f"🛑 [El Estratega - SÉCURITÉ DRIFT] Quarantaine critique active pour {symbol}. Ordre bloqué (NO-TRADE).")
            return {
                "action": "WAIT",
                "confidence": 0,
                "reason": reason,
                "cerebro_origen": "El Estratega",
                "is_quarantined": True,
            }

        # 0.1 Évaluation de sécurité macroéconomique préalable (El Economista)
        if economic_context and isinstance(economic_context, dict):
            is_blackout = bool(economic_context.get("high_impact_window", False))
            event_risk = str(economic_context.get("event_risk", "LOW")).upper()
            econ_rec = str(economic_context.get("recommendation", "NORMAL")).upper()
            econ_status = str(economic_context.get("status", "OK")).upper()

            if is_blackout or econ_rec == "NO_TRADE_NEWS" or event_risk in ("CRITICAL", "HIGH") or econ_status == "ERROR":
                reason = economic_context.get("reason", "Annonce macroéconomique majeure ou indisponibilité des données")
                logger.info(f"🛑 [El Estratega] Gel d'ordre macroéconomique imposé pour {symbol} : {reason}")
                return {
                    "action": "WAIT",
                    "confidence": 0,
                    "reason": f"Gel d'ordre macroéconomique (El Economista) : {reason}",
                    "economic_context": economic_context,
                    "cerebro_origen": "El Estratega",
                }

        # 1. Extraction si observation structurée d'El Observador est injectée
        if observation:
            tech_features = observation.get("tech_features", tech_features)
            amd_context = observation.get("amd", amd_context)
            if "regimen" in observation:
                htf_trend = observation.get("regimen_diagnostic", {}).get("bias", htf_trend)

        # 2. Évaluation de base par le DecisionEngine (Règles SMC + CRT + Boomerang + E(R))
        decision = self.decision_engine.evaluate(
            symbol=symbol,
            timeframe=timeframe,
            tech_features=tech_features,
            amd_context=amd_context,
            boomerang_data=boomerang_data,
            order_blocks=order_blocks,
            crt_setup=crt_setup,
            htf_trend=htf_trend,
            spread_price=spread_price,
            direction_filter=direction_filter,
        )

        # 3. Arbitrage de l'avis de l'IA externe si présent
        if ai_advice and isinstance(ai_advice, dict) and "action" in ai_advice:
            ai_action = str(ai_advice.get("action", "WAIT")).upper()
            local_action = str(decision.get("action", "WAIT")).upper()
            ai_conf = float(ai_advice.get("confidence", 0) or 0)
            local_conf = float(decision.get("confidence", 0) or 0)

            # Si l'IA externe propose une direction mais que le moteur quantitatif détecte une espérance négative ou WAIT
            if ai_action in ("BUY", "SELL") and local_action == "WAIT":
                # Vérification stricte : si l'espérance mathématique n'est pas positive, El Estratega bloque
                exp_r = float(decision.get("expected_value_r", 0.0) or 0.0)
                if exp_r <= 0:
                    decision["action"] = "WAIT"
                    decision["reason"] = (
                        f"Arbitrage El Estratega : Rejet de la proposition IA {ai_action} "
                        f"(Espérance E(R) = {exp_r:.2f}R <= 0, pas d'avantage statistique confirmé)."
                    )
                elif ai_conf >= 75:
                    # Confluence acceptée si l'IA a une forte confiance et qu'aucun filtre dur n'est violé
                    decision["action"] = ai_action
                    decision["confidence"] = round((local_conf + ai_conf) / 2.0, 1)
                    if "entry" in ai_advice and ai_advice.get("entry"):
                        decision["entry"] = ai_advice["entry"]
                        decision["sl"] = ai_advice.get("sl", decision.get("sl"))
                        decision["tp"] = ai_advice.get("tp", decision.get("tp"))
                    decision["reason"] = f"Confluence El Estratega + IA ({ai_advice.get('reason', '')})"

            elif ai_action in ("BUY", "SELL") and local_action in ("BUY", "SELL"):
                if ai_action == local_action:
                    # Double confirmation : bonus de confiance
                    decision["confidence"] = min(98.0, max(local_conf, ai_conf) + 5.0)
                    decision["reason"] = f"Alignement parfait Stratège & IA : {decision.get('reason')}"
                else:
                    # Conflit direct : RÈGLE DE SÉCURITÉ ABSOLUE = NO-TRADE (WAIT)
                    decision["action"] = "WAIT"
                    decision["confidence"] = 0
                    decision["reason"] = (
                        f"Conflit d'arbitrage : Moteur quantitatif propose {local_action} "
                        f"tandis que l'IA propose {ai_action}. Sécurité prioritaire = WAIT."
                    )

        # 4. Modération ou renforcement par le contexte macroéconomique (El Economista)
        if economic_context and isinstance(economic_context, dict):
            decision["economic_context"] = economic_context
            econ_bias = economic_context.get("bias", "NEUTRAL")
            current_action = decision.get("action")
            current_conf = float(decision.get("confidence", 50.0) or 50.0)

            if current_action == "BUY":
                if econ_bias == "BEARISH":
                    decision["confidence"] = max(10.0, current_conf - 15.0)
                    decision["reason"] += " (Attention: Contre-courant macroéconomique baissier)."
                elif econ_bias == "BULLISH":
                    decision["confidence"] = min(98.0, current_conf + 5.0)
                    decision["reason"] += " [Confluence macroéconomique haussière validée]."
            elif current_action == "SELL":
                if econ_bias == "BULLISH":
                    decision["confidence"] = max(10.0, current_conf - 15.0)
                    decision["reason"] += " (Attention: Contre-courant macroéconomique haussier)."
                elif econ_bias == "BEARISH":
                    decision["confidence"] = min(98.0, current_conf + 5.0)
                    decision["reason"] += " [Confluence macroéconomique baissière validée]."

        decision["cerebro_origen"] = "El Estratega"
        return decision

    def decidir_cuantitativo_puro(
        self,
        symbol: str,
        timeframe: str,
        tech_features: Dict[str, Any],
        amd_context: Optional[Dict[str, Any]] = None,
        boomerang_data: Optional[Dict[str, Any]] = None,
        order_blocks: Optional[List[Any]] = None,
        htf_trend: str = "Unknown",
        spread_price: float = 0.0,
        direction_filter: str = "BOTH",
        crt_setup: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Décision ultra-rapide (<5ms) via le LocalQuantBrain avec calibration logistique.
        """
        res = self.local_brain.decide(
            symbol=symbol,
            timeframe=timeframe,
            tech_features=tech_features,
            amd_context=amd_context,
            boomerang_data=boomerang_data,
            order_blocks=order_blocks,
            htf_trend=htf_trend,
            spread_price=spread_price,
            direction_filter=direction_filter,
            crt_setup=crt_setup,
        )
        res["cerebro_origen"] = "El Estratega"
        return res
