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
import math
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class LocalQuantBrain:
    def __init__(self, min_confidence: float = 65.0, min_rr: float = 1.5, weights_engine=None):
        self.min_confidence = min_confidence
        self.min_rr = min_rr
        self.weights_engine = weights_engine

    @staticmethod
    def _is_valid_numeric(val: Any, strictly_positive: bool = False) -> bool:
        """Vérifie la validité numérique stricte d'un indicateur technique."""
        if val is None:
            return False
        try:
            f = float(val)
            if math.isnan(f) or math.isinf(f):
                return False
            if strictly_positive and f <= 0:
                return False
            return True
        except (TypeError, ValueError):
            return False

    @classmethod
    def detect_market_regime(cls, tech_features: Dict[str, Any], amd_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Détermine le régime de marché déterministe actuel :
        - TRENDING_BULLISH : Tendance haussière affirmée (flux acheteur)
        - TRENDING_BEARISH : Tendance baissière affirmée (flux vendeur)
        - VOLATILE_EXPANSION : Forte volatilité / expansion de range
        - RANGE_CONSOLIDATION : Absence de tendance claire / consolidation
        - DEGRADED_DATA : Données techniques manquantes ou corrompues (Fail-Closed)
        """
        if not tech_features or not isinstance(tech_features, dict):
            return "DEGRADED_DATA"

        close_raw = tech_features.get("close")
        ema50_raw = tech_features.get("ema_50")
        ema200_raw = tech_features.get("ema_200")
        rsi_raw = tech_features.get("rsi_14")

        # Validation Fail-Closed : aucune valeur inventée par défaut
        if not (
            cls._is_valid_numeric(close_raw, strictly_positive=True)
            and cls._is_valid_numeric(ema50_raw, strictly_positive=True)
            and cls._is_valid_numeric(ema200_raw, strictly_positive=True)
            and cls._is_valid_numeric(rsi_raw)
        ):
            return "DEGRADED_DATA"

        amd_context = amd_context or {}
        close = float(close_raw)
        ema50 = float(ema50_raw)
        ema200 = float(ema200_raw)
        rsi = float(rsi_raw)
        adx = float(tech_features.get("adx") or 0.0)
        trend = str(tech_features.get("trend") or "NEUTRAL").upper()
        phase = str(amd_context.get("Phase", "")).upper()

        if "EXPANSION" in phase or "DISTRIBUTION" in phase or adx > 35.0:
            return "VOLATILE_EXPANSION"
        if (ema50 > ema200 and close > ema50) or "BULL" in trend:
            if adx >= 20.0 or abs(close - ema50) > 0.0005:
                return "TRENDING_BULLISH"
        if (ema50 < ema200 and close < ema50) or "BEAR" in trend:
            if adx >= 20.0 or abs(close - ema50) > 0.0005:
                return "TRENDING_BEARISH"
        return "RANGE_CONSOLIDATION"

    @staticmethod
    def calibrate_probability(technical_score: float, market_regime: str = "RANGE_CONSOLIDATION") -> float:
        """
        Modèle logistique calibrant le score technique brut en probabilité indicative P(Win).
        NOTE MÉTHODOLOGIQUE (Phase 1) :
        Cette valeur est une estimation heuristique sigmoïde bornée [0.35, 0.68].
        Elle porte le statut 'UNCALIBRATED_HEURISTIC' jusqu'à validation empirique hors-échantillon.
        """
        s = max(0.0, min(100.0, float(technical_score)))
        p_min = 0.35
        p_max = 0.65
        if "TRENDING" in market_regime:
            p_max = 0.68  # Léger avantage statistique dans le sens du trend
        elif market_regime == "VOLATILE_EXPANSION":
            p_max = 0.60  # Risque de faux signaux accru en pic de volatilité
        elif market_regime == "DEGRADED_DATA":
            return 0.0

        # Sigmoïde centrée sur 65.0 (seuil standard de déclenchement)
        sigmoid = 1.0 / (1.0 + math.exp(-0.07 * (s - 65.0)))
        calibrated_prob = p_min + (p_max - p_min) * sigmoid
        return round(calibrated_prob, 4)

    @staticmethod
    def calculate_expected_value_r(win_prob: float, reward_risk_ratio: float) -> float:
        """
        Calcule l'espérance mathématique en unités R :
        E(R) = P(Win) * RR - (1 - P(Win)) * 1.0
        """
        loss_prob = 1.0 - win_prob
        return round((win_prob * reward_risk_ratio) - (loss_prob * 1.0), 3)

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
        crt_setup: Any = None,
    ) -> Dict[str, Any]:
        """
        Génère une décision de trading quantitative complète, déterministe et vérifiée.
        Applique le principe FAIL-CLOSED : toute donnée technique manquante ou incohérente
        conduit à un refus d'ordre immédiat (WAIT).
        """
        amd_context = amd_context or {}
        boomerang_data = boomerang_data or {}
        order_blocks = order_blocks or []

        # 0. CONTRÔLE FAIL-CLOSED : Validation stricte des données techniques sans substitution arbitraire
        if not tech_features or not isinstance(tech_features, dict):
            inv_reason = "tech_features absent ou invalide"
            return {
                "action": "WAIT",
                "confidence": 0,
                "technical_score": 0.0,
                "statistical_win_prob": 0.0,
                "is_calibrated": False,
                "probability_status": "UNCALIBRATED_HEURISTIC",
                "expected_value_r": 0.0,
                "market_regime": "DEGRADED_DATA",
                "invalidation_reason": inv_reason,
                "entry": 0.0,
                "sl": 0.0,
                "tp": 0.0,
                "reason": f"Cerveau Quant Local [FAIL-CLOSED] : {inv_reason}",
                "journal_record": {
                    "symbol": symbol, "timeframe": timeframe, "action": "WAIT",
                    "entry": 0.0, "sl": 0.0, "tp": 0.0, "rr": 0.0,
                    "technical_score": 0.0, "statistical_win_prob": 0.0,
                    "is_calibrated": False, "probability_status": "UNCALIBRATED_HEURISTIC",
                    "expected_value_r": 0.0, "market_regime": "DEGRADED_DATA",
                    "confluences": [], "invalidation_reason": inv_reason,
                }
            }

        close_raw = tech_features.get("close")
        atr_raw = tech_features.get("atr_14")
        ema50_raw = tech_features.get("ema_50")
        ema200_raw = tech_features.get("ema_200")
        rsi_raw = tech_features.get("rsi_14")

        is_valid_input = (
            self._is_valid_numeric(close_raw, strictly_positive=True)
            and self._is_valid_numeric(atr_raw, strictly_positive=True)
            and self._is_valid_numeric(ema50_raw, strictly_positive=True)
            and self._is_valid_numeric(ema200_raw, strictly_positive=True)
            and self._is_valid_numeric(rsi_raw)
        )

        if not is_valid_input:
            safe_close = float(close_raw) if self._is_valid_numeric(close_raw) else 0.0
            inv_reason = "Données techniques manquantes, non finies ou invalides (RSI/EMA/ATR/Close) - Fail-Closed"
            return {
                "action": "WAIT",
                "confidence": 0,
                "technical_score": 0.0,
                "statistical_win_prob": 0.0,
                "is_calibrated": False,
                "probability_status": "UNCALIBRATED_HEURISTIC",
                "expected_value_r": 0.0,
                "market_regime": "DEGRADED_DATA",
                "invalidation_reason": inv_reason,
                "entry": safe_close,
                "sl": 0.0,
                "tp": 0.0,
                "reason": f"Cerveau Quant Local [FAIL-CLOSED] : {inv_reason}",
                "journal_record": {
                    "symbol": symbol, "timeframe": timeframe, "action": "WAIT",
                    "entry": safe_close, "sl": 0.0, "tp": 0.0, "rr": 0.0,
                    "technical_score": 0.0, "statistical_win_prob": 0.0,
                    "is_calibrated": False, "probability_status": "UNCALIBRATED_HEURISTIC",
                    "expected_value_r": 0.0, "market_regime": "DEGRADED_DATA",
                    "confluences": [], "invalidation_reason": inv_reason,
                }
            }

        close = float(close_raw)
        atr = float(atr_raw)
        ema50 = float(ema50_raw)
        ema200 = float(ema200_raw)
        rsi = float(rsi_raw)
        macd_hist = float(tech_features.get("macd_hist") or 0.0)
        trend = str(tech_features.get("trend") or "NEUTRAL").upper()

        market_regime = self.detect_market_regime(tech_features, amd_context)

        # ─── 1. FILTRES ANTI-CHASING / ANTI-FOMO ─────────────────────────────
        ema_dist_atr = (close - ema50) / atr
        if rsi >= 70 or ema_dist_atr > 2.5:
            # Surachat extrême ou cours trop étendu
            reason_msg = f"Cerveau Local: Marché en surachat sévère (RSI={rsi:.1f}, +{ema_dist_atr:.1f} ATR). Attente de repli."
            calibrated_p = self.calibrate_probability(0.0, market_regime)
            return {
                "action": "WAIT",
                "confidence": 85,
                "technical_score": 0.0,
                "statistical_win_prob": calibrated_p,
                "is_calibrated": False,
                "probability_status": "UNCALIBRATED_HEURISTIC",
                "expected_value_r": 0.0,
                "market_regime": market_regime,
                "invalidation_reason": "Anti-Chasing: Surachat extrême",
                "entry": close,
                "sl": 0.0,
                "tp": 0.0,
                "reason": reason_msg,
                "journal_record": {
                    "symbol": symbol, "timeframe": timeframe, "action": "WAIT",
                    "entry": close, "sl": 0.0, "tp": 0.0, "rr": 0.0,
                    "technical_score": 0.0, "statistical_win_prob": calibrated_p,
                    "is_calibrated": False, "probability_status": "UNCALIBRATED_HEURISTIC",
                    "expected_value_r": 0.0, "market_regime": market_regime,
                    "confluences": [], "invalidation_reason": "Anti-Chasing: Surachat extrême",
                }
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

        # Poids maximal disponible pour la normalisation rigoureuse
        crt_max_weight = 30.0 if crt_setup else 0.0
        total_available_weight = w_ob + w_boom + w_trend + w_rsi + w_htf + w_amd + crt_max_weight

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
        if ("EXPANSION" in phase or "DISTRIBUTION" in phase) and fvg != "None":
            if ema50 > ema200:
                bull_score += w_amd
            elif ema50 < ema200:
                bear_score += w_amd

        # G) Candle Range Theory (CRT) Setup
        if crt_setup:
            crt_dir = getattr(crt_setup, 'direction', 'NONE')
            kz_bonus = 5.0 if getattr(crt_setup, 'in_killzone', False) else 0.0
            w_crt = 25.0 + kz_bonus
            if crt_dir == "BUY":
                bull_score += w_crt
                reasons.append(f"CRT Bullish validé (Sweep C1 Low {crt_setup.c1_low:.5f}, close in range, EQ {crt_setup.c1_mid:.5f}, pds {w_crt:.0f})")
            elif crt_dir == "SELL":
                bear_score += w_crt
                reasons.append(f"CRT Bearish validé (Sweep C1 High {crt_setup.c1_high:.5f}, close in range, EQ {crt_setup.c1_mid:.5f}, pds {w_crt:.0f})")

        # ─── 3. DÉCISION ET CALCUL DES NIVEAUX DE PRIX ────────────────────────
        df_policy = str(direction_filter or "BOTH").upper()
        if df_policy == "SELL_ONLY":
            bull_score = 0.0
        elif df_policy == "BUY_ONLY":
            bear_score = 0.0

        min_sl_dist = max(atr * 1.0, min_distance, spread_price * 4)

        if bull_score >= self.min_confidence and bull_score > bear_score:
            entry = close
            # Stop loss placé sous la mèche CRT, sous l'Order Block ou sous le swing bas récent
            if crt_setup and getattr(crt_setup, 'direction', '') == "BUY":
                if float(crt_setup.sl_raw) >= entry:
                    sl = float(crt_setup.sl_raw)
                else:
                    sl = min(float(crt_setup.sl_raw) - (spread_price * 2 + min_distance), round(entry - min_sl_dist, 5))
                tp = float(crt_setup.tp2)
            elif active_ob and active_ob.direction == "BUY" and active_ob.sl:
                if float(active_ob.sl) >= entry:
                    sl = float(active_ob.sl)
                else:
                    sl = min(float(active_ob.sl), round(entry - min_sl_dist, 5))
                sl_dist = abs(entry - sl)
                tp = round(entry + sl_dist * 2.0, 5)
            else:
                sl = round(entry - max(atr * 1.5, min_sl_dist), 5)
                sl_dist = abs(entry - sl)
                tp = round(entry + sl_dist * 2.0, 5)

            tech_score = round(bull_score, 1)
            normalized_pct = min(100.0, round((bull_score / max(total_available_weight, 1.0)) * 100.0, 1))
            win_prob = self.calibrate_probability(tech_score, market_regime)

            # Contrôle de cohérence géométrique SL / TP
            if sl >= entry or tp <= entry:
                inv_reason = f"Géométrie des prix invalide pour BUY (entry={entry}, sl={sl}, tp={tp}). Ordre rejeté."
                calibrated_p = self.calibrate_probability(0.0, market_regime)
                return {
                    "action": "WAIT",
                    "confidence": 0,
                    "technical_score": tech_score,
                    "normalized_confluence_pct": normalized_pct,
                    "statistical_win_prob": calibrated_p,
                    "is_calibrated": False,
                    "probability_status": "UNCALIBRATED_HEURISTIC",
                    "expected_value_r": 0.0,
                    "market_regime": market_regime,
                    "invalidation_reason": inv_reason,
                    "entry": entry,
                    "sl": 0.0,
                    "tp": 0.0,
                    "reason": f"Cerveau Quant Local [FAIL-CLOSED] : {inv_reason}",
                    "journal_record": {
                        "symbol": symbol, "timeframe": timeframe, "action": "WAIT",
                        "entry": entry, "sl": 0.0, "tp": 0.0, "rr": 0.0,
                        "technical_score": tech_score, "normalized_confluence_pct": normalized_pct,
                        "statistical_win_prob": calibrated_p,
                        "is_calibrated": False, "probability_status": "UNCALIBRATED_HEURISTIC",
                        "expected_value_r": 0.0, "market_regime": market_regime,
                        "confluences": reasons, "invalidation_reason": inv_reason,
                    }
                }

            sl_dist = abs(entry - sl)
            tp_dist = abs(tp - entry)
            rr_ratio = round(tp_dist / max(sl_dist, 1e-9), 2)

            # Contrôle du ratio Risque/Rendement (1.0 minimum pour cibles structurelles CRT, self.min_rr sinon)
            min_rr_threshold = 1.0 if crt_setup else self.min_rr
            if rr_ratio < min_rr_threshold:
                inv_reason = f"Ratio R:R insuffisant ({rr_ratio:.2f} < {min_rr_threshold:.2f}). Ordre rejeté."
                calibrated_p = self.calibrate_probability(0.0, market_regime)
                return {
                    "action": "WAIT",
                    "confidence": 0,
                    "technical_score": tech_score,
                    "normalized_confluence_pct": normalized_pct,
                    "statistical_win_prob": calibrated_p,
                    "is_calibrated": False,
                    "probability_status": "UNCALIBRATED_HEURISTIC",
                    "expected_value_r": 0.0,
                    "market_regime": market_regime,
                    "invalidation_reason": inv_reason,
                    "entry": entry,
                    "sl": 0.0,
                    "tp": 0.0,
                    "reason": f"Cerveau Quant Local [FAIL-CLOSED] : {inv_reason}",
                    "journal_record": {
                        "symbol": symbol, "timeframe": timeframe, "action": "WAIT",
                        "entry": entry, "sl": 0.0, "tp": 0.0, "rr": 0.0,
                        "technical_score": tech_score, "normalized_confluence_pct": normalized_pct,
                        "statistical_win_prob": calibrated_p,
                        "is_calibrated": False, "probability_status": "UNCALIBRATED_HEURISTIC",
                        "expected_value_r": 0.0, "market_regime": market_regime,
                        "confluences": reasons, "invalidation_reason": inv_reason,
                    }
                }

            ev_r = self.calculate_expected_value_r(win_prob, rr_ratio)

            return {
                "action": "BUY",
                "confidence": min(95, int(bull_score)),
                "technical_score": tech_score,
                "normalized_confluence_pct": normalized_pct,
                "statistical_win_prob": win_prob,
                "is_calibrated": False,
                "probability_status": "UNCALIBRATED_HEURISTIC",
                "expected_value_r": ev_r,
                "market_regime": market_regime,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "reason": "Cerveau Quant Local : " + " + ".join(reasons),
                "journal_record": {
                    "symbol": symbol, "timeframe": timeframe, "action": "BUY",
                    "entry": entry, "sl": sl, "tp": tp, "rr": rr_ratio,
                    "technical_score": tech_score, "normalized_confluence_pct": normalized_pct,
                    "statistical_win_prob": win_prob,
                    "is_calibrated": False, "probability_status": "UNCALIBRATED_HEURISTIC",
                    "expected_value_r": ev_r, "market_regime": market_regime,
                    "confluences": reasons, "invalidation_reason": None,
                }
            }

        elif bear_score >= self.min_confidence and bear_score > bull_score:
            entry = close
            # Stop loss placé au-dessus de la mèche CRT, au-dessus de l'Order Block ou au-dessus du swing haut
            if crt_setup and getattr(crt_setup, 'direction', '') == "SELL":
                if float(crt_setup.sl_raw) <= entry:
                    sl = float(crt_setup.sl_raw)
                else:
                    sl = max(float(crt_setup.sl_raw) + (spread_price * 2 + min_distance), round(entry + min_sl_dist, 5))
                tp = float(crt_setup.tp2)
            elif active_ob and active_ob.direction == "SELL" and active_ob.sl:
                if float(active_ob.sl) <= entry:
                    sl = float(active_ob.sl)
                else:
                    sl = max(float(active_ob.sl), round(entry + min_sl_dist, 5))
                sl_dist = abs(sl - entry)
                tp = round(entry - sl_dist * 2.0, 5)
            else:
                sl = round(entry + max(atr * 1.5, min_sl_dist), 5)
                sl_dist = abs(sl - entry)
                tp = round(entry - sl_dist * 2.0, 5)

            tech_score = round(bear_score, 1)
            normalized_pct = min(100.0, round((bear_score / max(total_available_weight, 1.0)) * 100.0, 1))
            win_prob = self.calibrate_probability(tech_score, market_regime)

            # Contrôle de cohérence géométrique SL / TP
            if sl <= entry or tp >= entry:
                inv_reason = f"Géométrie des prix invalide pour SELL (entry={entry}, sl={sl}, tp={tp}). Ordre rejeté."
                calibrated_p = self.calibrate_probability(0.0, market_regime)
                return {
                    "action": "WAIT",
                    "confidence": 0,
                    "technical_score": tech_score,
                    "normalized_confluence_pct": normalized_pct,
                    "statistical_win_prob": calibrated_p,
                    "is_calibrated": False,
                    "probability_status": "UNCALIBRATED_HEURISTIC",
                    "expected_value_r": 0.0,
                    "market_regime": market_regime,
                    "invalidation_reason": inv_reason,
                    "entry": entry,
                    "sl": 0.0,
                    "tp": 0.0,
                    "reason": f"Cerveau Quant Local [FAIL-CLOSED] : {inv_reason}",
                    "journal_record": {
                        "symbol": symbol, "timeframe": timeframe, "action": "WAIT",
                        "entry": entry, "sl": 0.0, "tp": 0.0, "rr": 0.0,
                        "technical_score": tech_score, "normalized_confluence_pct": normalized_pct,
                        "statistical_win_prob": calibrated_p,
                        "is_calibrated": False, "probability_status": "UNCALIBRATED_HEURISTIC",
                        "expected_value_r": 0.0, "market_regime": market_regime,
                        "confluences": reasons, "invalidation_reason": inv_reason,
                    }
                }

            sl_dist = abs(sl - entry)
            tp_dist = abs(entry - tp)
            rr_ratio = round(tp_dist / max(sl_dist, 1e-9), 2)

            # Contrôle du ratio Risque/Rendement (1.0 minimum pour cibles structurelles CRT, self.min_rr sinon)
            min_rr_threshold = 1.0 if crt_setup else self.min_rr
            if rr_ratio < min_rr_threshold:
                inv_reason = f"Ratio R:R insuffisant ({rr_ratio:.2f} < {min_rr_threshold:.2f}). Ordre rejeté."
                calibrated_p = self.calibrate_probability(0.0, market_regime)
                return {
                    "action": "WAIT",
                    "confidence": 0,
                    "technical_score": tech_score,
                    "normalized_confluence_pct": normalized_pct,
                    "statistical_win_prob": calibrated_p,
                    "is_calibrated": False,
                    "probability_status": "UNCALIBRATED_HEURISTIC",
                    "expected_value_r": 0.0,
                    "market_regime": market_regime,
                    "invalidation_reason": inv_reason,
                    "entry": entry,
                    "sl": 0.0,
                    "tp": 0.0,
                    "reason": f"Cerveau Quant Local [FAIL-CLOSED] : {inv_reason}",
                    "journal_record": {
                        "symbol": symbol, "timeframe": timeframe, "action": "WAIT",
                        "entry": entry, "sl": 0.0, "tp": 0.0, "rr": 0.0,
                        "technical_score": tech_score, "normalized_confluence_pct": normalized_pct,
                        "statistical_win_prob": calibrated_p,
                        "is_calibrated": False, "probability_status": "UNCALIBRATED_HEURISTIC",
                        "expected_value_r": 0.0, "market_regime": market_regime,
                        "confluences": reasons, "invalidation_reason": inv_reason,
                    }
                }

            ev_r = self.calculate_expected_value_r(win_prob, rr_ratio)

            return {
                "action": "SELL",
                "confidence": min(95, int(bear_score)),
                "technical_score": tech_score,
                "normalized_confluence_pct": normalized_pct,
                "statistical_win_prob": win_prob,
                "is_calibrated": False,
                "probability_status": "UNCALIBRATED_HEURISTIC",
                "expected_value_r": ev_r,
                "market_regime": market_regime,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "reason": "Cerveau Quant Local : " + " + ".join(reasons),
                "journal_record": {
                    "symbol": symbol, "timeframe": timeframe, "action": "SELL",
                    "entry": entry, "sl": sl, "tp": tp, "rr": rr_ratio,
                    "technical_score": tech_score, "normalized_confluence_pct": normalized_pct,
                    "statistical_win_prob": win_prob,
                    "is_calibrated": False, "probability_status": "UNCALIBRATED_HEURISTIC",
                    "expected_value_r": ev_r, "market_regime": market_regime,
                    "confluences": reasons, "invalidation_reason": None,
                }
            }

        best_score = round(max(bull_score, bear_score), 1)
        normalized_pct = min(100.0, round((best_score / max(total_available_weight, 1.0)) * 100.0, 1))
        win_prob = self.calibrate_probability(best_score, market_regime)
        inv_reason = (
            f"Score de confluence insuffisant pour un avantage statistique avéré "
            f"(Score max={best_score:.1f} < Seuil min={self.min_confidence:.1f}) en régime {market_regime}."
        )

        return {
            "action": "WAIT",
            "confidence": 50,
            "technical_score": best_score,
            "normalized_confluence_pct": normalized_pct,
            "statistical_win_prob": win_prob,
            "is_calibrated": False,
            "probability_status": "UNCALIBRATED_HEURISTIC",
            "expected_value_r": 0.0,
            "market_regime": market_regime,
            "invalidation_reason": inv_reason,
            "entry": close,
            "sl": 0.0,
            "tp": 0.0,
            "reason": f"Cerveau Quant Local: {inv_reason}",
            "journal_record": {
                "symbol": symbol, "timeframe": timeframe, "action": "WAIT",
                "entry": close, "sl": 0.0, "tp": 0.0, "rr": 0.0,
                "technical_score": best_score, "normalized_confluence_pct": normalized_pct,
                "statistical_win_prob": win_prob,
                "is_calibrated": False, "probability_status": "UNCALIBRATED_HEURISTIC",
                "expected_value_r": 0.0, "market_regime": market_regime,
                "confluences": reasons, "invalidation_reason": inv_reason,
            }
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
