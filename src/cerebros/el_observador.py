# -*- coding: utf-8 -*-
"""
CEREBRO 1 : EL OBSERVADOR (Pilier 1 - OBSERVER)
================================================
Nom Espagnol : El Observador ("L'Observateur")
Rôle : Observe en continu la microstructure, la dynamique des prix et les régimes de marché.

Composants fédérés :
- MarketRegimeDetector : Qualification du régime (Trending, Range, Volatile Expansion, Choppy Illiquide)
- AMDEngine : Cycles de manipulation institutionnelle (Accumulation, Manipulation, Distribution)
- OrderFlowEngine : Microstructure L2, CVD (Cumulative Volume Delta), Volume Profile et absorption
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd

from src.core.market_regime import MarketRegimeDetector, MarketRegimeType
from src.core.amd_detector import AMDEngine
from src.core.order_flow_engine import OrderFlowEngine

logger = logging.getLogger(__name__)


class ElObservador:
    """
    Cerveau Observateur (El Observador)
    Scanne, filtre et analyse les conditions de marché avant toute prise de décision.
    """

    def __init__(
        self,
        regime_detector: Optional[MarketRegimeDetector] = None,
        amd_engine: Optional[AMDEngine] = None,
        order_flow_engine: Optional[OrderFlowEngine] = None,
        mt5_connector=None,
    ):
        self.mt5 = mt5_connector
        self.regime_detector = regime_detector or MarketRegimeDetector(adx_trending_threshold=22.0)
        self.amd_engine = amd_engine or AMDEngine(lookback_period=20)
        self.order_flow_engine = order_flow_engine or OrderFlowEngine()

    def detectar_regimen(
        self,
        tech_features: Dict[str, Any],
        amd_context: Optional[Dict[str, Any]] = None,
        spread_ratio: float = 0.0,
    ) -> Dict[str, Any]:
        """Détecte le régime de marché déterministe (Trending Bull/Bear, Range, Volatile, Choppy)."""
        return self.regime_detector.detect_regime(tech_features, amd_context=amd_context, spread_ratio=spread_ratio)

    def analizar_amd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse les cycles institutionnels d'Accumulation, Manipulation et Distribution."""
        if df is None or len(df) < 20:
            return {
                "Phase": "Unknown",
                "FVG": "None",
                "BOS_MSS": "None",
                "Liquidity_Grab": "None",
            }
        return {
            "Phase": self.amd_engine.detect_amd_phase(df),
            "FVG": self.amd_engine.detect_fvg(df),
            "Liquidity_Grab": self.amd_engine.detect_liquidity_grab(df),
            "BOS_MSS": self.amd_engine.detect_bos_mss(df),
        }

    def observar_mercado(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        tech_features: Dict[str, Any],
        ticks_df: Optional[pd.DataFrame] = None,
        dom_data: Optional[Dict[str, Any]] = None,
        spread_price: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Génère une observation holistique complète du marché pour El Estratega.
        """
        amd = self.analizar_amd(df)
        atr_val = float(tech_features.get("atr_14") or 0.001)
        spread_ratio = (spread_price / atr_val) if atr_val > 0 else 0.0
        regime = self.detectar_regimen(tech_features, amd_context=amd, spread_ratio=spread_ratio)

        # Microstructure et Order Flow
        of_confluence = self.order_flow_engine.evaluate_setup_confluence(
            symbol=symbol,
            crt_setup=None,
            ticks_df=ticks_df,
            dom_data=dom_data,
            candles_df=df,
            atr=atr_val,
        )

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "regimen": regime.get("regime", MarketRegimeType.RANGE_CONSOLIDATION),
            "regimen_diagnostic": regime,
            "amd": amd,
            "order_flow": of_confluence,
            "tech_features": tech_features,
            "es_hachado_o_toxico": regime.get("regime") == MarketRegimeType.CHOPPY_ILLIQUID,
        }
