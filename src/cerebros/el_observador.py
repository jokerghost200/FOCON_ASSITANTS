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

    def evaluar_killzone_horaria(
        self,
        utc_hour: Optional[int] = None,
        utc_minute: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Évalue la fenêtre horaire institutionnelle (Killzones SMC/CRT) :
        - Killzone Londres (07:00 - 11:00 UTC) : +6% confiance
        - Killzone New York / Overlap (12:30 - 16:30 UTC) : +6% confiance
        - Creux de Rollover Minuit (21:00 - 22:59 UTC) : -15% confiance (spreads élargis / faux signaux)
        """
        import datetime
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        h = now_utc.hour if utc_hour is None else int(utc_hour)
        m = now_utc.minute if utc_minute is None else int(utc_minute)
        hm = h + (m / 60.0)

        if 7.0 <= hm <= 11.0:
            return {
                "status": "KILLZONE_ACTIVE",
                "session": "LONDON_KILLZONE",
                "score_bonus": 6.0,
                "label": "🏆 Killzone Londres (Fort Volume Institutionnel)",
            }
        if 12.5 <= hm <= 16.5:
            return {
                "status": "KILLZONE_ACTIVE",
                "session": "NY_OVERLAP_KILLZONE",
                "score_bonus": 6.0,
                "label": "🏆 Killzone New York / Overlap (Liquidité Max SMC/CRT)",
            }
        if 21.0 <= hm < 23.0:
            return {
                "status": "ROLLOVER_BLOCK",
                "session": "ROLLOVER_DEADZONE",
                "score_bonus": -15.0,
                "label": "🚫 Creux de Rollover Bancaire (Spreads Élargis / Faible Liquidité)",
            }
        return {
            "status": "NORMAL",
            "session": "STANDARD",
            "score_bonus": 0.0,
            "label": "📊 Session Standard",
        }

    def analizar_zona_premium_discount(self, df: pd.DataFrame, lookback: int = 50) -> Dict[str, Any]:
        """
        Analyse le Dealing Range institutionnel (Premium vs Discount Fibonacci 50% & OTE 61.8%-78.6%) :
        - Discount (< 50%, OTE 15%-38.2%) : zone d'achat institutionnelle idéale
        - Premium (> 50%, OTE 61.8%-85%) : zone de vente institutionnelle idéale
        """
        if df is None or len(df) < 15:
            return {"zone": "EQUILIBRIUM", "pct_in_range": 50.0, "buy_mod": 0.0, "sell_mod": 0.0, "label": "Équilibre 50%"}

        sub = df.tail(min(len(df), lookback))
        r_high = float(sub["high"].max())
        r_low = float(sub["low"].min())
        close = float(sub["close"].iloc[-1])
        span = r_high - r_low
        if span <= 1e-9:
            return {"zone": "EQUILIBRIUM", "pct_in_range": 50.0, "buy_mod": 0.0, "sell_mod": 0.0, "label": "Équilibre 50%"}

        pct = ((close - r_low) / span) * 100.0
        eq_50 = r_low + span * 0.50

        if 15.0 <= pct <= 38.2:
            return {
                "zone": "OTE_DISCOUNT",
                "pct_in_range": round(pct, 1),
                "equilibrium": eq_50,
                "buy_mod": 6.0,
                "sell_mod": -8.0,
                "label": f"🎯 Zone OTE Discount ({pct:.0f}% du range — Achat Institutionnel Optimal)",
            }
        if pct < 50.0:
            return {
                "zone": "DISCOUNT",
                "pct_in_range": round(pct, 1),
                "equilibrium": eq_50,
                "buy_mod": 3.0,
                "sell_mod": -5.0,
                "label": f"🟢 Zone Discount ({pct:.0f}% < 50% — Favorable BUY)",
            }
        if 61.8 <= pct <= 85.0:
            return {
                "zone": "OTE_PREMIUM",
                "pct_in_range": round(pct, 1),
                "equilibrium": eq_50,
                "buy_mod": -8.0,
                "sell_mod": 6.0,
                "label": f"🎯 Zone OTE Premium ({pct:.0f}% du range — Vente Institutionnelle Optimale)",
            }
        return {
            "zone": "PREMIUM",
            "pct_in_range": round(pct, 1),
            "equilibrium": eq_50,
            "buy_mod": -5.0,
            "sell_mod": 3.0,
            "label": f"🔴 Zone Premium ({pct:.0f}% > 50% — Favorable SELL)",
        }

    def detectar_piscinas_liquidez(self, df: pd.DataFrame, atr: float = 0.0) -> Dict[str, Any]:
        """
        Détecte les piscines de liquidité institutionnelles :
        - Plus Haut / Plus Bas de la session précédente (PDH / PDL)
        - Equal Highs (EQH) / Equal Lows (EQL)
        - Balayage récent (Liquidity Sweep + réintégration) sur les 3 dernières bougies
        """
        if df is None or len(df) < 25:
            return {"sweep": "NONE", "buy_bonus": 0.0, "sell_bonus": 0.0, "label": "Aucun sweep majeur"}

        n = len(df)
        prev_window = df.iloc[max(0, n - 96):max(5, n - 3)]
        recent_3 = df.tail(3)

        pdh = float(prev_window["high"].max())
        pdl = float(prev_window["low"].min())
        last_close = float(recent_3["close"].iloc[-1])
        recent_max_h = float(recent_3["high"].max())
        recent_min_l = float(recent_3["low"].min())

        # Détection de Sweep PDL (mèche sous le plus bas précédent + clôture au-dessus)
        if recent_min_l < pdl and last_close > pdl:
            return {
                "sweep": "SWEEP_PDL_BULLISH",
                "pdh": pdh,
                "pdl": pdl,
                "buy_bonus": 8.0,
                "sell_bonus": -8.0,
                "label": "🧲 Sweep Liquidité Basse (PDL/EQL balayé + réintégration haussière)",
            }

        # Détection de Sweep PDH (mèche au-dessus du plus haut précédent + clôture en-dessous)
        if recent_max_h > pdh and last_close < pdh:
            return {
                "sweep": "SWEEP_PDH_BEARISH",
                "pdh": pdh,
                "pdl": pdl,
                "buy_bonus": -8.0,
                "sell_bonus": 8.0,
                "label": "🧲 Sweep Liquidité Haute (PDH/EQH balayé + réintégration baissière)",
            }

        return {
            "sweep": "NONE",
            "pdh": pdh,
            "pdl": pdl,
            "buy_bonus": 0.0,
            "sell_bonus": 0.0,
            "label": "Liquidité standard",
        }

    def analizar_volumen_vsa(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyse Volume Spread Analysis (VSA) & Volume Relatif Tick (RVOL) :
        Mesure l'activité institutionnelle réelle sur la bougie actuelle par rapport aux 20 dernières bougies.
        """
        if df is None or len(df) < 20 or "tick_volume" not in df.columns:
            return {"rvol": 1.0, "state": "NORMAL", "buy_bonus": 0.0, "sell_bonus": 0.0, "label": "Volume standard"}

        vols = df["tick_volume"].tail(20).astype(float)
        avg_vol = float(vols.iloc[:-1].mean() or 1.0)
        last_vol = float(vols.iloc[-1] or avg_vol)
        rvol = last_vol / max(1.0, avg_vol)

        last = df.iloc[-1]
        h, l, o, c = float(last["high"]), float(last["low"]), float(last["open"]), float(last["close"])
        rng = max(1e-9, h - l)
        lower_wick = (min(o, c) - l) / rng
        upper_wick = (h - max(o, c)) / rng

        if rvol >= 1.4 and lower_wick >= 0.35 and c >= o:
            return {
                "rvol": round(rvol, 2),
                "state": "BULLISH_ABSORPTION",
                "buy_bonus": 5.0,
                "sell_bonus": -4.0,
                "label": f"📈 Absorption Acheteuse VSA (RVOL {rvol:.1f}x + rejet bas)",
            }
        if rvol >= 1.4 and upper_wick >= 0.35 and c <= o:
            return {
                "rvol": round(rvol, 2),
                "state": "BEARISH_ABSORPTION",
                "buy_bonus": -4.0,
                "sell_bonus": 5.0,
                "label": f"📉 Absorption Vendeuse VSA (RVOL {rvol:.1f}x + rejet haut)",
            }
        if rvol < 0.45:
            return {
                "rvol": round(rvol, 2),
                "state": "LOW_VOLUME_WEAK",
                "buy_bonus": -4.0,
                "sell_bonus": -4.0,
                "label": f"⚠️ Volume Tick Faible (RVOL {rvol:.2f}x — manque d'engagement)",
            }
        return {
            "rvol": round(rvol, 2),
            "state": "NORMAL",
            "buy_bonus": 2.0 if (rvol >= 1.3 and c > o) else 0.0,
            "sell_bonus": 2.0 if (rvol >= 1.3 and c < o) else 0.0,
            "label": f"Volume RVOL {rvol:.1f}x",
        }

    def evaluar_alineacion_mtf(
        self,
        m15_trend: str = "Unknown",
        h1_trend: str = "Unknown",
        h4_trend: str = "Unknown",
    ) -> Dict[str, Any]:
        """
        Calcule le Score d'Alignement Fractal Multi-Timeframe (M15 + H1 + H4).
        """
        t_m15 = str(m15_trend or "").upper()
        t_h1 = str(h1_trend or "").upper()
        t_h4 = str(h4_trend or "").upper()

        bull_count = sum(1 for t in (t_m15, t_h1, t_h4) if "BULL" in t)
        bear_count = sum(1 for t in (t_m15, t_h1, t_h4) if "BEAR" in t)

        if bull_count == 3:
            return {
                "alignment": "FULL_BULLISH",
                "buy_bonus": 6.0,
                "sell_bonus": -10.0,
                "label": "🌊 Alignement Fractal Complet M15+H1+H4 HAUSSIER",
            }
        if bear_count == 3:
            return {
                "alignment": "FULL_BEARISH",
                "buy_bonus": -10.0,
                "sell_bonus": 6.0,
                "label": "🌊 Alignement Fractal Complet M15+H1+H4 BAISSIER",
            }
        if "BULL" in t_h1 and "BEAR" in t_h4:
            return {
                "alignment": "CONFLICT_H1_H4",
                "buy_bonus": -3.0,
                "sell_bonus": -3.0,
                "label": "⚖️ Transition H1/H4 contradictoire (Prudence)",
            }
        if "BEAR" in t_h1 and "BULL" in t_h4:
            return {
                "alignment": "CONFLICT_H1_H4",
                "buy_bonus": -3.0,
                "sell_bonus": -3.0,
                "label": "⚖️ Transition H1/H4 contradictoire (Prudence)",
            }
        return {
            "alignment": "STANDARD",
            "buy_bonus": 2.0 if bull_count >= 2 else 0.0,
            "sell_bonus": 2.0 if bear_count >= 2 else 0.0,
            "label": "Alignement MTF standard",
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
        htf_trend: str = "Unknown",
        h4_trend: str = "Unknown",
    ) -> Dict[str, Any]:
        """
        Génère une observation holistique complète du marché pour El Estratega :
        - Régime de marché & ADX
        - Cycles AMD (Accumulation, Manipulation, Distribution, FVG, BOS/MSS)
        - Zone Premium / Discount (Fibonacci 50% & OTE)
        - Piscines de liquidité (PDH/PDL & Equal Highs/Lows Sweeps)
        - Volume Spread Analysis (RVOL & bougies d'absorption)
        - Alignement Fractal Multi-Timeframe (M15 + H1 + H4)
        - Killzones horaires institutionnelles
        """
        amd = self.analizar_amd(df)
        atr_val = float(tech_features.get("atr_14") or 0.001)
        spread_ratio = (spread_price / atr_val) if atr_val > 0 else 0.0
        regime = self.detectar_regimen(tech_features, amd_context=amd, spread_ratio=spread_ratio)
        killzone = self.evaluar_killzone_horaria()
        premium_discount = self.analizar_zona_premium_discount(df)
        liquidity_pools = self.detectar_piscinas_liquidez(df, atr=atr_val)
        vsa_volume = self.analizar_volumen_vsa(df)
        mtf_alignment = self.evaluar_alineacion_mtf(
            m15_trend=str(tech_features.get("trend", "Unknown")),
            h1_trend=htf_trend,
            h4_trend=h4_trend,
        )

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
            "killzone": killzone,
            "premium_discount": premium_discount,
            "liquidity_pools": liquidity_pools,
            "vsa_volume": vsa_volume,
            "mtf_alignment": mtf_alignment,
            "es_hachado_o_toxico": regime.get("regime") == MarketRegimeType.CHOPPY_ILLIQUID,
        }
