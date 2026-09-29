"""
Local Quantitative Trading Brain (Cerveau Quantitatif Local)
============================================================
Moteur de décision algorithmique 100% autonome et hors-ligne.
Fonctionne sans aucune requête API externe (Gemini/OpenRouter).
Utilisé comme système de secours instantané en cas d'erreur API,
de dépassement de quota (HTTP 429), de panne réseau ou de latence.

Stratégie :
1. Smart Money Concepts (Order Blocks institutionnels retestés)
2. Boomerang Breakout-Retest-Rejet
3. Alignement des moyennes mobiles exponentielles (EMA 50 / 200)
4. Filtres de repli stochastique / RSI (anti-chasing / anti-sommet)
5. Alignement multi-timeframe (HTF Trend)
"""

import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class LocalQuantBrain:
    def __init__(self, min_confidence: float = 65.0, min_rr: float = 1.5, weights_engine=None):
        self.min_confidence = min_confidence
        self.min_rr = min_rr
        self.weights_engine = weights_engine

    def decide(
        self,
        symbol: str,
        timeframe: str,
        tech_features: Dict[str, Any],
        amd_context: Dict[str, Any] = None,
        boomerang_data: Dict[str, Any] = None,
        order_blocks: List[Any] = None,
        htf_trend: str = "Unknown",
        spread_price: float = 0.0,
        min_distance: float = 0.0001,
        direction_filter: str = "BOTH",
    ) -> Dict[str, Any]:
        """
        Génère une décision de trading quantitative complète et déterministe.
        Retourne un dictionnaire compatible avec le format de décision de l'IA :
        {
            "action": "BUY" | "SELL" | "WAIT",
            "confidence": int (0-100),
            "entry": float,
            "sl": float,
            "tp": float,
            "reason": str
        }
        """
        amd_context = amd_context or {}
        boomerang_data = boomerang_data or {}
        order_blocks = order_blocks or []

        close = float(tech_features.get("close") or 0.0)
        ema50 = float(tech_features.get("ema_50") or close)
        ema200 = float(tech_features.get("ema_200") or close)
        rsi = float(tech_features.get("rsi_14") or 50.0)
        atr = float(tech_features.get("atr_14") or (close * 0.001))
        macd_hist = float(tech_features.get("macd_hist") or 0.0)
        trend = str(tech_features.get("trend") or "NEUTRAL").upper()

        if close <= 0 or atr <= 0:
            return {
                "action": "WAIT",
                "confidence": 0,
                "entry": close,
                "sl": close,
                "tp": close,
                "reason": "Cerveau Local: Données techniques insuffisantes",
            }

        # ─── 1. FILTRES ANTI-CHASING / ANTI-FOMO ─────────────────────────────
        ema_dist_atr = (close - ema50) / atr
        if rsi >= 70 or ema_dist_atr > 2.5:
            # Surachat extrême ou cours trop étendu
            return {
                "action": "WAIT",
                "confidence": 85,
                "entry": close,
                "sl": 0.0,
                "tp": 0.0,
                "reason": f"Cerveau Local: Marché en surachat sévère (RSI={rsi:.1f}, +{ema_dist_atr:.1f} ATR). Attente de repli.",
            }
        if rsi <= 30 or ema_dist_atr < -2.5:
            # Survente extrême ou cours trop étendu
            return {
                "action": "WAIT",
                "confidence": 85,
                "entry": close,
                "sl": 0.0,
                "tp": 0.0,
                "reason": f"Cerveau Local: Marché en survente sévère (RSI={rsi:.1f}, {ema_dist_atr:.1f} ATR). Attente de rebond.",
            }

        # ─── 2. ANALYSE DES CONFLUENCES INSTITUTIONNELLES & POIDS ADAPTATIFS ────
        weights = self.weights_engine.get_weights() if self.weights_engine else {
            "ob_weight": 35.0,
            "boomerang_weight": 30.0,
            "trend_ema_weight": 15.0,
            "rsi_pullback_weight": 15.0,
            "htf_alignment_weight": 15.0,
            "amd_expansion_weight": 10.0,
        }
        w_ob = float(weights.get("ob_weight", 35.0))
        w_boom = float(weights.get("boomerang_weight", 30.0))
        w_trend = float(weights.get("trend_ema_weight", 15.0))
        w_rsi = float(weights.get("rsi_pullback_weight", 15.0))
        w_htf = float(weights.get("htf_alignment_weight", 15.0))
        w_amd = float(weights.get("amd_expansion_weight", 10.0))

        bull_score = 0.0
        bear_score = 0.0
        reasons = []

        # A) Order Blocks retestés (SMC)
        active_ob = None
        for ob in order_blocks:
            if getattr(ob, "is_retested", False) and getattr(ob, "score", 0) >= 50:
                active_ob = ob
                break

        if active_ob:
            if active_ob.direction == "BUY":
                bull_score += w_ob
                reasons.append(f"OB Bullish retesté ({active_ob.ob_type}, score {active_ob.score:.0f}, pds {w_ob:.0f})")
            elif active_ob.direction == "SELL":
                bear_score += w_ob
                reasons.append(f"OB Bearish retesté ({active_ob.ob_type}, score {active_ob.score:.0f}, pds {w_ob:.0f})")

        # B) Boomerang Setup
        boom_decision = boomerang_data.get("decision", "INVALID")
        boom_dir = boomerang_data.get("direction", "NONE")
        boom_score = float(boomerang_data.get("score", 0))

        if boom_decision == "VALID" and boom_score >= 60:
            if boom_dir == "BUY":
                bull_score += w_boom
                reasons.append(f"Boomerang VALID BUY ({boom_score:.0f}/100, pds {w_boom:.0f})")
            elif boom_dir == "SELL":
                bear_score += w_boom
                reasons.append(f"Boomerang VALID SELL ({boom_score:.0f}/100, pds {w_boom:.0f})")

        # C) Tendance des moyennes mobiles & Pullback
        if ema50 > ema200 and close >= ema50:
            bull_score += w_trend
            reasons.append(f"Tendance haussière EMA50>200 (pds {w_trend:.0f})")
        elif ema50 < ema200 and close <= ema50:
            bear_score += w_trend
            reasons.append(f"Tendance baissière EMA50<200 (pds {w_trend:.0f})")

        # D) RSI dans la zone de repli saine
        if 42.0 <= rsi <= 60.0 and ema50 > ema200:
            bull_score += w_rsi
            reasons.append(f"Repli sain support RSI {rsi:.1f} (pds {w_rsi:.0f})")
        elif 40.0 <= rsi <= 58.0 and ema50 < ema200:
            bear_score += w_rsi
            reasons.append(f"Rebond sain résistance RSI {rsi:.1f} (pds {w_rsi:.0f})")

        # E) Alignement Higher Timeframe (HTF)
        htf_str = str(htf_trend).upper()
        if "BULL" in htf_str:
            bull_score += w_htf
            bear_score -= 20.0
        elif "BEAR" in htf_str:
            bear_score += w_htf
            bull_score -= 20.0

        # F) AMD SMC Phase
        phase = str(amd_context.get("Phase", "")).upper()
        fvg = str(amd_context.get("FVG", "None"))
        if "EXPANSION" in phase and fvg != "None":
            if ema50 > ema200:
                bull_score += w_amd
            elif ema50 < ema200:
                bear_score += w_amd

        # ─── 3. DÉCISION ET CALCUL DES NIVEAUX DE PRIX ────────────────────────
        df_policy = str(direction_filter or "BOTH").upper()
        if df_policy == "SELL_ONLY":
            bull_score = 0.0
        elif df_policy == "BUY_ONLY":
            bear_score = 0.0

        min_sl_dist = max(atr * 1.0, min_distance, spread_price * 4)

        if bull_score >= self.min_confidence and bull_score > bear_score:
            entry = close
            # Stop loss placé sous l'Order Block ou sous le swing bas récent
            if active_ob and active_ob.direction == "BUY" and active_ob.sl:
                sl = min(float(active_ob.sl), round(entry - min_sl_dist, 5))
            else:
                sl = round(entry - max(atr * 1.5, min_sl_dist), 5)

            sl_dist = abs(entry - sl)
            tp = round(entry + sl_dist * max(self.min_rr, 2.0), 5)

            return {
                "action": "BUY",
                "confidence": min(95, int(bull_score)),
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "reason": "Cerveau Quant Local : " + " + ".join(reasons),
            }

        elif bear_score >= self.min_confidence and bear_score > bull_score:
            entry = close
            # Stop loss placé au-dessus de l'Order Block ou au-dessus du swing haut récent
            if active_ob and active_ob.direction == "SELL" and active_ob.sl:
                sl = max(float(active_ob.sl), round(entry + min_sl_dist, 5))
            else:
                sl = round(entry + max(atr * 1.5, min_sl_dist), 5)

            sl_dist = abs(sl - entry)
            tp = round(entry - sl_dist * max(self.min_rr, 2.0), 5)

            return {
                "action": "SELL",
                "confidence": min(95, int(bear_score)),
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "reason": "Cerveau Quant Local : " + " + ".join(reasons),
            }

        return {
            "action": "WAIT",
            "confidence": 50,
            "entry": close,
            "sl": 0.0,
            "tp": 0.0,
            "reason": "Cerveau Quant Local: En attente d'une confluence claire (pas de setup SMC/Breakout suffisant).",
        }

    def evaluate_position(self, pos: Dict[str, Any], tech_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Évalue localement une position ouverte sans appel externe.
        """
        profit = float(pos.get("profit") or 0.0)
        pos_type = str(pos.get("type") or "BUY").upper()
        rsi = float(tech_features.get("rsi_14") or 50.0)
        trend = str(tech_features.get("trend") or "").upper()

        # Si le trade est en profit et que les indicateurs indiquent un retournement
        if profit >= 3.0:
            if pos_type == "BUY" and (rsi >= 75.0 or "BEAR" in trend):
                return {
                    "action": "CLOSE",
                    "reason": f"Cerveau Local: Prise de profit protectrice (Gain={profit:.2f}$, RSI suracheté {rsi:.1f} ou tendance inversée)",
                }
            elif pos_type == "SELL" and (rsi <= 25.0 or "BULL" in trend):
                return {
                    "action": "CLOSE",
                    "reason": f"Cerveau Local: Prise de profit protectrice (Gain={profit:.2f}$, RSI survendu {rsi:.1f} ou tendance inversée)",
                }

        return {
            "action": "HOLD",
            "reason": "Cerveau Local: Position saine, maintien du trade.",
        }
