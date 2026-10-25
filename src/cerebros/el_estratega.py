# -*- coding: utf-8 -*-
"""
CEREBRO 2 : EL ESTRATEGA (Pilier 2 - RÉFLÉCHIR & DÉCIDER)
=========================================================
Nom Espagnol : El Estratega ("Le Stratège")
Rôle : Formule, évalue et sélectionne les meilleures décisions de trading
en combinant les setups institutionnels, la calibration logistique et l'espérance mathématique.

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
    Cerveau Stratège (El Estratega)
    Analyse les confluences, applique les règles de gestion institutionnelles
    et produit des signaux rigoureux à espérance mathématique positive.
    """

    def __init__(
        self,
        local_brain: Optional[LocalQuantBrain] = None,
        decision_engine: Optional[DecisionEngine] = None,
        observador=None,
        weights_engine=None,
    ):
        self.weights_engine = weights_engine
        self.local_brain = local_brain or LocalQuantBrain(min_confidence=60.0, min_rr=1.5, weights_engine=weights_engine)
        self.decision_engine = decision_engine or DecisionEngine(
            local_brain=self.local_brain,
            regime_detector=getattr(observador, "regime_detector", None),
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
    ) -> Dict[str, Any]:
        """
        Génère une décision stratégique complète par arbitrage institutionnel.
        """
        # Utilise le DecisionEngine pour évaluer et calibrer
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
        return self.local_brain.decide(
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
