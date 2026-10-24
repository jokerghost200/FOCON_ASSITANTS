"""
FATI Market Regime Engine (Détecteur de Régime de Marché Autonome)
==================================================================
Identifie de façon déterministe le régime macro et micro de marché :
- TRENDING_BULLISH : Flux acheteur aligné, momentum fort
- TRENDING_BEARISH : Flux vendeur aligné, momentum fort
- RANGE_CONSOLIDATION : Absence de tendance, compression de volatilité
- VOLATILE_EXPANSION : Forte expansion ATR, pic de volatilité
- CHOPPY_ILLIQUID : Marché désordonné ou spread excessif (abstention)
"""

from enum import Enum
from typing import Dict, Any, Optional
import math
import logging

logger = logging.getLogger(__name__)


class MarketRegimeType(str, Enum):
    TRENDING_BULLISH = "TRENDING_BULLISH"
    TRENDING_BEARISH = "TRENDING_BEARISH"
    RANGE_CONSOLIDATION = "RANGE_CONSOLIDATION"
    VOLATILE_EXPANSION = "VOLATILE_EXPANSION"
    CHOPPY_ILLIQUID = "CHOPPY_ILLIQUID"


class MarketRegimeDetector:
    """
    Détecteur autonome de régime de marché.
    Analyse l'alignement des moyennes mobiles, la force de tendance (ADX),
    la compression/expansion (ATR & Bollinger) et la liquidité (spread/ATR).
    """

    def __init__(self, adx_trending_threshold: float = 22.0, high_vol_atr_mult: float = 1.6):
        self.adx_trending_threshold = adx_trending_threshold
        self.high_vol_atr_mult = high_vol_atr_mult

    def detect_regime(
        self,
        tech_features: Dict[str, Any],
        amd_context: Optional[Dict[str, Any]] = None,
        spread_ratio: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Détermine le régime de marché et renvoie un diagnostic complet :
        {
            "regime": MarketRegimeType,
            "confidence": float (0-100),
            "is_tradeable": bool,
            "trend_strength": float,
            "volatility_state": str,
            "reason": str
        }
        """
        tech = tech_features or {}
        amd = amd_context or {}

        close = float(tech.get("close") or 0.0)
        ema50 = float(tech.get("ema_50") or close)
        ema200 = float(tech.get("ema_200") or close)
        rsi = float(tech.get("rsi_14") or 50.0)
        atr = float(tech.get("atr_14") or 0.001)
        adx = float(tech.get("adx") or 0.0)
        trend_label = str(tech.get("trend") or "NEUTRAL").upper()
        phase = str(amd.get("Phase", "")).upper()

        # 1. Filtre d'illiquidité ou spread toxique
        if spread_ratio > 0.35:
            return {
                "regime": MarketRegimeType.CHOPPY_ILLIQUID,
                "confidence": 90.0,
                "is_tradeable": False,
                "trend_strength": adx,
                "volatility_state": "ILLIQUID",
                "reason": f"Marché illiquide ou coût de spread excessif ({spread_ratio*100:.1f}% ATR). Abstention requise.",
            }

        # 2. Pic de volatilité / Expansion explosive
        ema_dist_atr = abs(close - ema50) / max(atr, 1e-9)
        if "EXPANSION" in phase or "DISTRIBUTION" in phase or adx >= 40.0 or ema_dist_atr > 3.0:
            return {
                "regime": MarketRegimeType.VOLATILE_EXPANSION,
                "confidence": 85.0,
                "is_tradeable": True,
                "trend_strength": adx,
                "volatility_state": "HIGH_EXPANSION",
                "reason": f"Forte expansion de volatilité (ADX={adx:.1f}, dist EMA={ema_dist_atr:.1f} ATR). Préférer breakouts et réjections.",
            }

        # 3. Tendance directionnelle affirmée
        is_bull_align = (ema50 > ema200 and close >= ema50) or "BULL" in trend_label
        is_bear_align = (ema50 < ema200 and close <= ema50) or "BEAR" in trend_label

        if is_bull_align and adx >= self.adx_trending_threshold:
            return {
                "regime": MarketRegimeType.TRENDING_BULLISH,
                "confidence": min(95.0, 60.0 + (adx - 20.0) * 1.5),
                "is_tradeable": True,
                "trend_strength": adx,
                "volatility_state": "HEALTHY_TREND",
                "reason": f"Tendance haussière saine (EMA50>200, ADX={adx:.1f}, RSI={rsi:.1f}). Privilégier les pullbacks acheteurs.",
            }

        if is_bear_align and adx >= self.adx_trending_threshold:
            return {
                "regime": MarketRegimeType.TRENDING_BEARISH,
                "confidence": min(95.0, 60.0 + (adx - 20.0) * 1.5),
                "is_tradeable": True,
                "trend_strength": adx,
                "volatility_state": "HEALTHY_TREND",
                "reason": f"Tendance baissière saine (EMA50<200, ADX={adx:.1f}, RSI={rsi:.1f}). Privilégier les rebonds vendeurs.",
            }

        # 4. Marché en range ou consolidation
        return {
            "regime": MarketRegimeType.RANGE_CONSOLIDATION,
            "confidence": 75.0,
            "is_tradeable": True,
            "trend_strength": adx,
            "volatility_state": "NORMAL_RANGE",
            "reason": f"Consolidation latérale (ADX={adx:.1f} sous seuil, EMA plates). Privilégier les balayages de liquidité (CRT / SMC).",
        }
