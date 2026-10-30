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

    @staticmethod
    def es_simbolo_crypto(symbol: str) -> bool:
        """Détecte si un symbole est une cryptomonnaie (marché 24/7 actif le week-end)."""
        if not symbol:
            return False
        sym_u = str(symbol).upper()
        crypto_roots = (
            "BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE", "AVAX",
            "DOT", "LTC", "LINK", "LNK", "XLM", "BCH", "MATIC", "SHIB", "NEAR", "UNI", "TON", "TRX",
        )
        return any(sym_u.startswith(c) or c in sym_u for c in crypto_roots)

    @staticmethod
    def es_fin_de_semana(now_utc=None) -> bool:
        """Vérifie si nous sommes dans la fenêtre Week-End (Vendredi >= 21h00 UTC à Dimanche < 22h00 UTC)."""
        import datetime
        dt = now_utc or datetime.datetime.now(datetime.timezone.utc)
        wd = dt.weekday()  # 4=Ven, 5=Sam, 6=Dim
        if wd == 4 and dt.hour >= 21:
            return True
        if wd == 5:
            return True
        if wd == 6 and dt.hour < 22:
            return True
        return False

    def evaluar_killzone_horaria(
        self,
        utc_hour: Optional[int] = None,
        utc_minute: Optional[int] = None,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Évalue la fenêtre horaire institutionnelle (Killzones SMC/CRT) :
        - Killzone Londres (07:00 - 11:00 UTC) : +6% confiance
        - Killzone New York / Overlap (12:30 - 16:30 UTC) : +6% confiance
        - Creux de Rollover Minuit (21:00 - 22:59 UTC) : -15% confiance sur Forex uniquement
        - Sur Crypto 24/7 : pas de rollover bancaire (sauf réouverture CME dimanche 21h-23h UTC)
        """
        import datetime
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        h = now_utc.hour if utc_hour is None else int(utc_hour)
        m = now_utc.minute if utc_minute is None else int(utc_minute)
        hm = h + (m / 60.0)
        is_crypto = self.es_simbolo_crypto(symbol or "")

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
            if is_crypto:
                if now_utc.weekday() == 6:
                    return {
                        "status": "CME_REOPEN_WATCH",
                        "session": "CME_REOPEN",
                        "score_bonus": -5.0,
                        "label": "⚡ Réouverture CME Crypto Dimanche Soir (Volatilité accrue)",
                    }
                return {
                    "status": "CRYPTO_24_7",
                    "session": "CRYPTO_CONTINUOUS",
                    "score_bonus": 2.0,
                    "label": "🪙 Session Crypto 24/7 (Pas de Rollover Bancaire)",
                }
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

    def analizar_estructura_zigzag_fractal(
        self,
        df: pd.DataFrame,
        tech_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyse la structure de marché via les indicateurs ZigZag (séquence HH/HL vs LH/LL)
        et Williams Fractals (Micro-BOS et rejets sur niveaux fractals).
        """
        tech = tech_features or {}
        if df is None or len(df) < 15:
            return {
                "structure": "NEUTRAL",
                "zigzag_trend": 0,
                "fractal_signal": "NONE",
                "buy_bonus": 0.0,
                "sell_bonus": 0.0,
                "label": "Structure ZigZag/Fractal neutre",
            }

        zz_trend = int(tech.get("zigzag_trend") or (df["zigzag_trend"].iloc[-1] if "zigzag_trend" in df.columns else 0) or 0)

        # 1. Analyse de la séquence des pivots ZigZag (HH / HL / LH / LL)
        zz_structure = "NEUTRAL"
        if "zigzag_pivot" in df.columns and "zigzag_value" in df.columns:
            piv_df = df[df["zigzag_pivot"].isin(["H", "L"])][["zigzag_pivot", "zigzag_value"]].dropna()
            highs_piv = piv_df[piv_df["zigzag_pivot"] == "H"]["zigzag_value"].values
            lows_piv = piv_df[piv_df["zigzag_pivot"] == "L"]["zigzag_value"].values

            has_hh = len(highs_piv) >= 2 and float(highs_piv[-1]) > float(highs_piv[-2])
            has_lh = len(highs_piv) >= 2 and float(highs_piv[-1]) < float(highs_piv[-2])
            has_hl = len(lows_piv) >= 2 and float(lows_piv[-1]) > float(lows_piv[-2])
            has_ll = len(lows_piv) >= 2 and float(lows_piv[-1]) < float(lows_piv[-2])

            if has_hh and has_hl:
                zz_structure = "BULLISH_HH_HL"
            elif has_lh and has_ll:
                zz_structure = "BEARISH_LH_LL"
            elif has_hh and has_ll and zz_trend > 0:
                zz_structure = "CHOCH_BULLISH"
            elif has_lh and has_hl and zz_trend < 0:
                zz_structure = "CHOCH_BEARISH"
        else:
            if zz_trend > 0:
                zz_structure = "ZIGZAG_UP"
            elif zz_trend < 0:
                zz_structure = "ZIGZAG_DOWN"

        # 2. Analyse des Fractales de Williams (Micro-BOS ou Rejet sur niveau fractal)
        fractal_signal = "NONE"
        last_close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else last_close
        last_high = float(df["high"].iloc[-1])
        last_low = float(df["low"].iloc[-1])

        frac_high = tech.get("last_fractal_high")
        frac_low = tech.get("last_fractal_low")
        if frac_high is None and "last_fractal_high" in df.columns and len(df) >= 2:
            val_h = df["last_fractal_high"].iloc[-2]
            frac_high = float(val_h) if pd.notnull(val_h) else None
        if frac_low is None and "last_fractal_low" in df.columns and len(df) >= 2:
            val_l = df["last_fractal_low"].iloc[-2]
            frac_low = float(val_l) if pd.notnull(val_l) else None

        if frac_high and frac_high > 0:
            if prev_close <= frac_high and last_close > frac_high:
                fractal_signal = "FRACTAL_BOS_UP"
            elif last_high > frac_high and last_close < frac_high:
                fractal_signal = "FRACTAL_SWEEP_BEAR"

        if fractal_signal == "NONE" and frac_low and frac_low > 0:
            if prev_close >= frac_low and last_close < frac_low:
                fractal_signal = "FRACTAL_BOS_DOWN"
            elif last_low < frac_low and last_close > frac_low:
                fractal_signal = "FRACTAL_SWEEP_BULL"

        # 3. Calcul des bonus / pénalités de confluence structurelle
        buy_bonus = 0.0
        sell_bonus = 0.0
        labels = []

        if zz_structure == "BULLISH_HH_HL":
            buy_bonus += 5.0
            sell_bonus -= 6.0
            labels.append("ZigZag HH+HL Haussier")
        elif zz_structure == "BEARISH_LH_LL":
            sell_bonus += 5.0
            buy_bonus -= 6.0
            labels.append("ZigZag LH+LL Baissier")
        elif zz_structure == "CHOCH_BULLISH":
            buy_bonus += 4.0
            labels.append("ZigZag CHoCH Haussier")
        elif zz_structure == "CHOCH_BEARISH":
            sell_bonus += 4.0
            labels.append("ZigZag CHoCH Baissier")
        elif zz_trend > 0:
            buy_bonus += 2.0
            labels.append("ZigZag ↑")
        elif zz_trend < 0:
            sell_bonus += 2.0
            labels.append("ZigZag ↓")

        if fractal_signal in ("FRACTAL_BOS_UP", "FRACTAL_SWEEP_BULL"):
            buy_bonus += 4.0
            sell_bonus -= 3.0
            labels.append("Fractal Bullish Confirmé" if fractal_signal == "FRACTAL_BOS_UP" else "Sweep Fractal Bas")
        elif fractal_signal in ("FRACTAL_BOS_DOWN", "FRACTAL_SWEEP_BEAR"):
            sell_bonus += 4.0
            buy_bonus -= 3.0
            labels.append("Fractal Bearish Confirmé" if fractal_signal == "FRACTAL_BOS_DOWN" else "Sweep Fractal Haut")

        return {
            "structure": zz_structure,
            "zigzag_trend": zz_trend,
            "fractal_signal": fractal_signal,
            "last_fractal_high": frac_high,
            "last_fractal_low": frac_low,
            "buy_bonus": round(buy_bonus, 1),
            "sell_bonus": round(sell_bonus, 1),
            "label": " + ".join(labels) if labels else "Structure ZigZag/Fractal neutre",
        }

    def analizar_contexto_crypto_weekend(
        self,
        df: pd.DataFrame,
        symbol: str,
        atr: float = 0.0,
        now_utc=None,
    ) -> Dict[str, Any]:
        """
        Cerveau Spécialiste Crypto Week-End ('El Criptógrafo') :
        - Actif sur les paires Crypto (BTCUSD, ETHUSD, SOLUSD, XRPUSD, etc.), avec règles renforcées le week-end :
          1. Aimant du CME Anchor (cours d'équilibre / clôture vendredi soir) : favorise le retour à l'équilibre
             après une extension > 2.0 ATR hors du CME Anchor.
          2. Détection des Chasses aux Liquidations (Whale Stop-Hunts) sur les extrêmes du week-end.
          3. Filtre Anti-Bull/Bear Trap : pénalise les cassures sèches aux extrêmes sans volume (RVOL < 1.25).
        """
        is_crypto = self.es_simbolo_crypto(symbol)
        if not is_crypto or df is None or len(df) < 25:
            return {
                "active": False,
                "is_crypto": is_crypto,
                "is_weekend": False,
                "pattern": "NONE",
                "cme_anchor": 0.0,
                "buy_bonus": 0.0,
                "sell_bonus": 0.0,
                "label": "",
            }

        is_wknd = self.es_fin_de_semana(now_utc)
        window = df.tail(min(64, len(df)))
        cme_anchor = float(window["close"].iloc[0])
        wk_high = float(window["high"].iloc[:-1].max())
        wk_low = float(window["low"].iloc[:-1].min())
        wk_range = max(wk_high - wk_low, 1e-9)

        last = df.iloc[-1]
        c = float(last["close"])
        o = float(last["open"])
        h = float(last["high"])
        l = float(last["low"])
        pos_pct = ((c - wk_low) / wk_range) * 100.0
        eff_atr = atr if atr > 0 else (wk_range / 10.0)
        anchor_dist_atr = (c - cme_anchor) / max(eff_atr, 1e-9)

        # RVOL
        rvol = 1.0
        if "tick_volume" in df.columns and len(df) >= 20:
            avg_v = float(df["tick_volume"].tail(21).iloc[:-1].mean() or 1.0)
            if avg_v > 0:
                rvol = float(last["tick_volume"]) / avg_v

        buy_bonus = 0.0
        sell_bonus = 0.0
        pattern = "CRYPTO_STANDARD"
        label = f"🪙 Crypto {'Week-End' if is_wknd else '24/7'}"

        # 1. Whale Stop-Hunt / Liquidation Sweep sur le bas ou le haut du range
        if l < wk_low and c > wk_low and c >= o:
            pattern = "WHALE_LONG_LIQUIDATION_SWEEP"
            buy_bonus += 9.0 if is_wknd else 6.0
            sell_bonus -= 10.0
            label = f"🐋 Crypto Liquidation Sweep Bas + Rejet ({wk_low:.2f})"
        elif h > wk_high and c < wk_high and c <= o:
            pattern = "WHALE_SHORT_LIQUIDATION_SWEEP"
            sell_bonus += 9.0 if is_wknd else 6.0
            buy_bonus -= 10.0
            label = f"🐋 Crypto Liquidation Sweep Haut + Rejet ({wk_high:.2f})"

        # 2. Aimant CME Gap / Mean Reversion le week-end quand le prix s'éloigne trop sans volume
        elif is_wknd:
            if pos_pct >= 82.0 and rvol < 1.25:
                pattern = "WEEKEND_BULL_TRAP_RISK"
                buy_bonus -= 12.0
                if c < o and anchor_dist_atr > 1.5:
                    sell_bonus += 6.0
                label = f"⚠️ Crypto Week-End Bull-Trap (Haut de range {pos_pct:.0f}% sans volume RVOL {rvol:.1f}x)"
            elif pos_pct <= 18.0 and rvol < 1.25:
                pattern = "WEEKEND_BEAR_TRAP_RISK"
                sell_bonus -= 12.0
                if c > o and anchor_dist_atr < -1.5:
                    buy_bonus += 6.0
                label = f"⚠️ Crypto Week-End Bear-Trap (Bas de range {pos_pct:.0f}% sans volume RVOL {rvol:.1f}x)"
            elif anchor_dist_atr <= -2.0 and c > o:
                pattern = "CME_ANCHOR_REVERSION_BUY"
                buy_bonus += 6.0
                label = f"🧲 Retour vers CME Anchor ({cme_anchor:.2f}, écart {anchor_dist_atr:+.1f} ATR)"
            elif anchor_dist_atr >= 2.0 and c < o:
                pattern = "CME_ANCHOR_REVERSION_SELL"
                sell_bonus += 6.0
                label = f"🧲 Retour vers CME Anchor ({cme_anchor:.2f}, écart {anchor_dist_atr:+.1f} ATR)"

        return {
            "active": True,
            "is_crypto": True,
            "is_weekend": is_wknd,
            "pattern": pattern,
            "cme_anchor": round(cme_anchor, 2),
            "anchor_dist_atr": round(anchor_dist_atr, 2),
            "rvol": round(rvol, 2),
            "buy_bonus": round(buy_bonus, 1),
            "sell_bonus": round(sell_bonus, 1),
            "label": label,
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
        - Structure ZigZag (HH/HL vs LH/LL) & Fractales de Williams
        - Cerveau Spécialiste Crypto Week-End (CME Anchor & Whale Liquidation Sweeps)
        - Killzones horaires institutionnelles
        """
        amd = self.analizar_amd(df)
        atr_val = float(tech_features.get("atr_14") or 0.001)
        spread_ratio = (spread_price / atr_val) if atr_val > 0 else 0.0
        regime = self.detectar_regimen(tech_features, amd_context=amd, spread_ratio=spread_ratio)
        killzone = self.evaluar_killzone_horaria(symbol=symbol)
        premium_discount = self.analizar_zona_premium_discount(df)
        liquidity_pools = self.detectar_piscinas_liquidez(df, atr=atr_val)
        vsa_volume = self.analizar_volumen_vsa(df)
        mtf_alignment = self.evaluar_alineacion_mtf(
            m15_trend=str(tech_features.get("trend", "Unknown")),
            h1_trend=htf_trend,
            h4_trend=h4_trend,
        )
        zigzag_fractal = self.analizar_estructura_zigzag_fractal(df, tech_features=tech_features)
        crypto_weekend = self.analizar_contexto_crypto_weekend(df, symbol=symbol, atr=atr_val)

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
            "zigzag_fractal": zigzag_fractal,
            "crypto_weekend": crypto_weekend,
            "es_hachado_o_toxico": regime.get("regime") == MarketRegimeType.CHOPPY_ILLIQUID,
        }
