"""
Order Block Detector
====================
Détecte les Order Blocks (OB) institutionnels selon les Smart Money Concepts (SMC).

Un Order Block est :
- La DERNIÈRE bougie bearish avant une forte impulsion bullish (Bullish OB = support d'achat)
- La DERNIÈRE bougie bullish avant une forte impulsion bearish (Bearish OB = résistance de vente)

Améliorations clés :
  1. Détection robuste des impulsions (bougie unique explosive OU impulsion sur 2 bougies).
  2. Filtrage d'invalidation : vérifie que l'OB n'a pas été brisé/traversé depuis sa création.
  3. Détection souple du retest :
     - Retest actif immédiat (cours actuel dans la zone OB ± tolérance)
     - OU retest récent avec mèche de rejet au cours des 1 à 3 dernières bougies.
  4. Niveaux précis (Entry, SL sous la zone + marge ATR, TP à R:R >= 2.0).
  5. Scoring SMC complet (fraîcheur, force d'impulsion, mèche de rejet, alignement HTF, taille de zone).
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)

OB_SCORE_VALID = 65
OB_IMPULSE_ATR = 1.6
OB_RETEST_TOLERANCE = 0.8


@dataclass
class OrderBlock:
    ob_type: str
    direction: str
    ob_high: float
    ob_low: float
    ob_mid: float
    bar_index: int
    bars_ago: int
    impulse_strength: float
    is_retested: bool = False
    has_rejection: bool = False
    score: float = 0.0
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    rr: Optional[float] = None
    reasons: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "RETESTE (ACTIF)" if self.is_retested else "EN ATTENTE"
        lines = [
            f"ORDER BLOCK {self.ob_type} | {self.direction}",
            f"Zone     : [{self.ob_low:.5f} - {self.ob_high:.5f}] (mid: {self.ob_mid:.5f})",
            f"Il y a   : {self.bars_ago} bougies",
            f"Impulsion: {self.impulse_strength:.1f}x ATR",
            f"Statut   : {status}",
            f"Score    : {self.score:.0f}/100",
        ]
        if self.entry:
            lines.append(f"Entry    : {self.entry:.5f} | SL: {self.sl:.5f} | TP: {self.tp:.5f} | R:R: {self.rr:.2f}")
        return "\n".join(lines)


class OrderBlockDetector:
    def __init__(
        self,
        impulse_atr_mult: float = OB_IMPULSE_ATR,
        lookback: int = 100,
        retest_tolerance: float = OB_RETEST_TOLERANCE,
        sl_buffer_mult: float = 0.3,
        tp_rr_mult: float = 2.5,
        min_sl_atr_mult: float = 0.8,
    ):
        self.impulse_atr_mult = impulse_atr_mult
        self.lookback = lookback
        self.retest_tolerance = retest_tolerance
        self.sl_buffer_mult = sl_buffer_mult
        self.tp_rr_mult = tp_rr_mult
        self.min_sl_atr_mult = min_sl_atr_mult

    def detect(self, df: pd.DataFrame, symbol: str, timeframe: str, htf_trend: str = "", direction_filter: str = "BOTH") -> List[OrderBlock]:
        if df is None or len(df) < 20:
            return []
        df = df.copy().reset_index(drop=True)
        if 'ATR_14' not in df.columns:
            df = self._compute_atr(df)
        atr = float(df['ATR_14'].iloc[-1])
        if atr <= 0:
            return []

        current_price = float(df['close'].iloc[-1])
        start = max(0, len(df) - self.lookback)
        window = df.iloc[start:]
        obs = []

        # 1. Détection des impulsions et des OB associés
        for i in range(1, len(window) - 1):
            abs_i = start + i
            row_curr = window.iloc[i]
            row_prev = window.iloc[i - 1]

            o_curr, c_curr = float(row_curr['open']), float(row_curr['close'])
            o_prev, c_prev = float(row_prev['open']), float(row_prev['close'])

            # --- Impulsion Bullish (Achat) ---
            if direction_filter in ("BUY", "BOTH"):
                single_bull = (c_curr > o_curr) and ((c_curr - o_curr) >= self.impulse_atr_mult * atr)
                double_bull = (c_curr > o_prev) and ((c_curr - o_prev) >= self.impulse_atr_mult * atr) and (c_curr > c_prev)

                if single_bull or double_bull:
                    ob = self._find_ob_before_impulse(df, abs_i, 'BUY', atr, current_price)
                    if ob and not self._is_ob_invalidated(df, ob, 'BUY', atr):
                        obs.append(ob)

            # --- Impulsion Bearish (Vente) ---
            if direction_filter in ("SELL", "BOTH"):
                single_bear = (c_curr < o_curr) and ((o_curr - c_curr) >= self.impulse_atr_mult * atr)
                double_bear = (c_curr < o_prev) and ((o_prev - c_curr) >= self.impulse_atr_mult * atr) and (c_curr < c_prev)

                if single_bear or double_bear:
                    ob = self._find_ob_before_impulse(df, abs_i, 'SELL', atr, current_price)
                    if ob and not self._is_ob_invalidated(df, ob, 'SELL', atr):
                        obs.append(ob)

        # 2. Déduplication
        obs = self._deduplicate(obs, atr)

        # 3. Scoring et calcul des niveaux
        for ob in obs:
            self._score_ob(ob, df, atr, htf_trend)
            if ob.is_retested:
                self._calc_levels(ob, atr, current_price)

        # 4. Sélection des OB valides (retestés et score suffisant)
        valid_obs = [ob for ob in obs if ob.is_retested and ob.score >= OB_SCORE_VALID]
        valid_obs.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"[OB] {symbol} {timeframe}: {len(valid_obs)} OB valides retestes sur {len(obs)} detectes")
        return valid_obs

    def _find_ob_before_impulse(self, df: pd.DataFrame, impulse_idx: int, direction: str, atr: float, current_price: float) -> Optional[OrderBlock]:
        tol = self.retest_tolerance * atr
        # Recherche la dernière bougie opposée avant l'impulsion (jusqu'à 7 bougies en arrière)
        for j in range(impulse_idx - 1, max(0, impulse_idx - 7), -1):
            row = df.iloc[j]
            o = float(row['open'])
            c = float(row['close'])
            h = float(row['high'])
            l = float(row['low'])
            bars_ago = len(df) - 1 - j

            impulse_row = df.iloc[impulse_idx]
            impulse_move = abs(float(impulse_row['close']) - float(impulse_row['open']))
            impulse_strength = impulse_move / atr

            if direction == 'BUY' and c <= o:
                # Bullish OB : dernière bougie baissière
                is_retested, has_rejection = self._check_retest(df, j, 'BUY', l, h, tol, current_price)
                return OrderBlock(
                    ob_type='Bullish OB',
                    direction='BUY',
                    ob_high=h,
                    ob_low=l,
                    ob_mid=(h + l) / 2,
                    bar_index=j,
                    bars_ago=bars_ago,
                    impulse_strength=impulse_strength,
                    is_retested=is_retested,
                    has_rejection=has_rejection,
                )
            elif direction == 'SELL' and c >= o:
                # Bearish OB : dernière bougie haussière
                is_retested, has_rejection = self._check_retest(df, j, 'SELL', l, h, tol, current_price)
                return OrderBlock(
                    ob_type='Bearish OB',
                    direction='SELL',
                    ob_high=h,
                    ob_low=l,
                    ob_mid=(h + l) / 2,
                    bar_index=j,
                    bars_ago=bars_ago,
                    impulse_strength=impulse_strength,
                    is_retested=is_retested,
                    has_rejection=has_rejection,
                )
        return None

    def _check_retest(self, df: pd.DataFrame, ob_idx: int, direction: str, ob_low: float, ob_high: float, tol: float, current_price: float):
        """
        Vérifie si le prix est en retest de la zone de l'OB :
        1. Soit le prix actuel est dans la zone élargie [ob_low - tol, ob_high + tol]
        2. Soit l'une des 3 dernières bougies a touché la zone et rejette dans le bon sens
        """
        in_zone_now = (ob_low - tol) <= current_price <= (ob_high + tol)

        # Vérification sur les 3 dernières bougies
        recent_window = df.iloc[max(ob_idx + 1, len(df) - 3):]
        retested_recently = False
        has_rejection = False

        if not recent_window.empty:
            for _, r in recent_window.iterrows():
                r_low, r_high = float(r['low']), float(r['high'])
                r_open, r_close = float(r['open']), float(r['close'])
                r_range = r_high - r_low
                if direction == 'BUY':
                    # A touché la zone et n'a pas clôturé en dessous
                    if r_low <= (ob_high + tol) and r_close >= (ob_low - tol):
                        retested_recently = True
                        # Rejet haussier réel : mèche basse >= 30% du range OU bougie verte clôturant au-dessus du milieu de l'OB
                        lower_wick = (r_close - r_low) if r_close < r_open else (r_open - r_low)
                        if (r_range > 0 and (lower_wick / r_range) >= 0.30) or (r_close > r_open and r_close > (ob_low + ob_high) / 2):
                            has_rejection = True
                else:  # SELL
                    # A touché la zone et n'a pas clôturé au-dessus
                    if r_high >= (ob_low - tol) and r_close <= (ob_high + tol):
                        retested_recently = True
                        # Rejet baissier réel : mèche haute >= 30% du range OU bougie rouge clôturant sous le milieu de l'OB
                        upper_wick = (r_high - r_close) if r_close > r_open else (r_high - r_open)
                        if (r_range > 0 and (upper_wick / r_range) >= 0.30) or (r_close < r_open and r_close < (ob_low + ob_high) / 2):
                            has_rejection = True

        is_retested = in_zone_now or retested_recently
        return is_retested, has_rejection

    def _is_ob_invalidated(self, df: pd.DataFrame, ob: OrderBlock, direction: str, atr: float) -> bool:
        """
        Un Order Block est invalidé si une bougie ultérieure clôture au-delà de la zone de sécurité (SL).
        - Pour un Bullish OB : clôture sous ob_low - (sl_buffer_mult * atr).
        - Pour un Bearish OB : clôture au-dessus de ob_high + (sl_buffer_mult * atr).
        """
        future_candles = df.iloc[ob.bar_index + 1:]
        if future_candles.empty:
            return False

        buf = self.sl_buffer_mult * atr
        if direction == 'BUY':
            return bool((future_candles['close'] < (ob.ob_low - buf)).any())
        else:
            return bool((future_candles['close'] > (ob.ob_high + buf)).any())

    def _score_ob(self, ob: OrderBlock, df: pd.DataFrame, atr: float, htf_trend: str):
        score = 0.0
        reasons = []

        # 1. Fraîcheur (max 30 pts) — au-delà de 60 bougies, pénalité sévère
        if ob.bars_ago > 60:
            freshness = 0.0
            reasons.append(f"OB trop ancien ({ob.bars_ago} bougies): 0pts")
        else:
            freshness = max(0.0, 30.0 - ob.bars_ago * 0.5)
            reasons.append(f"Fraîcheur: {freshness:.0f}pts ({ob.bars_ago} bougies)")
        score += freshness

        # 2. Force de l'impulsion (max 25 pts)
        impl_score = min(25.0, max(10.0, (ob.impulse_strength - self.impulse_atr_mult) * 10.0 + 15.0))
        score += impl_score
        reasons.append(f"Impulsion: {ob.impulse_strength:.1f}x ATR -> {impl_score:.0f}pts")

        # 3. Retest et rejet (max 30 pts)
        if ob.is_retested:
            retest_pts = 20.0
            if ob.has_rejection:
                retest_pts += 10.0
                reasons.append("Retest avec rejet: +30pts")
            else:
                reasons.append("Retest actif: +20pts")
            score += retest_pts

        # 4. Alignement de tendance HTF (max 20 pts)
        t = (htf_trend or '').upper()
        if (ob.direction == 'BUY' and 'BULL' in t) or (ob.direction == 'SELL' and 'BEAR' in t):
            score += 20.0
            reasons.append("HTF aligné: +20pts")
        elif (ob.direction == 'BUY' and 'BEAR' in t) or (ob.direction == 'SELL' and 'BULL' in t):
            score -= 25.0  # Rejet strict des OB à contre-courant
            reasons.append("HTF opposé: -25pts (Danger)")
        else:
            reasons.append("HTF neutre: 0pts")

        # 5. Taille saine de la zone OB (max 10 pts)
        zone_size = ob.ob_high - ob.ob_low
        if 0.2 * atr <= zone_size <= 2.5 * atr:
            score += 10.0
            reasons.append("Zone OB calibrée: +10pts")

        ob.score = min(100.0, max(0.0, score))
        ob.reasons = reasons

    def _calc_levels(self, ob: OrderBlock, atr: float, current_price: float):
        buf = self.sl_buffer_mult * atr
        # Plancher minimal de Stop Loss (minimum 8 pips) pour ne pas être liquidé par le spread
        point_est = 0.00001 if current_price < 20 else (0.001 if current_price < 500 else 0.01)
        min_pips_floor = 80 * point_est  # 8 pips
        min_sl_dist = max(self.min_sl_atr_mult * atr, 1.2 * atr, min_pips_floor)
        tol = self.retest_tolerance * atr

        if ob.direction == 'BUY':
            if (ob.ob_low - tol) <= current_price <= (ob.ob_high + tol):
                entry = current_price
            else:
                entry = (ob.ob_mid + ob.ob_low) / 2

            raw_sl = ob.ob_low - buf
            sl_dist = entry - raw_sl
            if sl_dist < min_sl_dist:
                raw_sl = entry - min_sl_dist
                sl_dist = min_sl_dist

            ob.entry = round(entry, 5)
            ob.sl = round(raw_sl, 5)
            ob.tp = round(ob.entry + sl_dist * self.tp_rr_mult, 5)
        else:  # SELL
            if (ob.ob_low - tol) <= current_price <= (ob.ob_high + tol):
                entry = current_price
            else:
                entry = (ob.ob_mid + ob.ob_high) / 2

            raw_sl = ob.ob_high + buf
            sl_dist = raw_sl - entry
            if sl_dist < min_sl_dist:
                raw_sl = entry + min_sl_dist
                sl_dist = min_sl_dist

            ob.entry = round(entry, 5)
            ob.sl = round(raw_sl, 5)
            ob.tp = round(ob.entry - sl_dist * self.tp_rr_mult, 5)

        sl_dist = abs(ob.entry - ob.sl)
        tp_dist = abs(ob.tp - ob.entry)
        ob.rr = round(tp_dist / sl_dist, 2) if sl_dist > 0 else None

    def _deduplicate(self, obs: List[OrderBlock], atr: float) -> List[OrderBlock]:
        unique = []
        for ob in sorted(obs, key=lambda x: x.bars_ago):
            overlap = any(kept.direction == ob.direction and abs(kept.ob_mid - ob.ob_mid) < (0.8 * atr) for kept in unique)
            if not overlap:
                unique.append(ob)
        return unique

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        df['ATR_14'] = tr.ewm(span=period, adjust=False).mean()
        return df

    def detect_pending_zones(
        self, df: pd.DataFrame, symbol: str, timeframe: str,
        htf_trend: str = "", direction_filter: str = "BOTH",
        min_pending_score: float = 50.0,
    ) -> List[OrderBlock]:
        """
        Détecte les Order Blocks de haute qualité NON ENCORE retestés.
        
        Ce sont des zones où le prix n'est PAS encore revenu, mais qui sont 
        statistiquement très probables comme zones de rebond institutionnel.
        Parfaites pour y placer des ordres limites (pending).
        
        Différence avec `detect()` :
        - `detect()` retourne les OB retestés (pour exécution immédiate)
        - `detect_pending_zones()` retourne les OB EN ATTENTE (pour pending orders)
        
        Returns
        -------
        List[OrderBlock] avec entry, sl, tp calculés, triés par score décroissant.
        """
        if df is None or len(df) < 20:
            return []
        df = df.copy().reset_index(drop=True)
        if 'ATR_14' not in df.columns:
            df = self._compute_atr(df)
        atr = float(df['ATR_14'].iloc[-1])
        if atr <= 0:
            return []

        current_price = float(df['close'].iloc[-1])
        start = max(0, len(df) - self.lookback)
        window = df.iloc[start:]
        obs = []

        for i in range(1, len(window) - 1):
            abs_i = start + i
            row_curr = window.iloc[i]
            row_prev = window.iloc[i - 1]

            o_curr, c_curr = float(row_curr['open']), float(row_curr['close'])
            o_prev, c_prev = float(row_prev['open']), float(row_prev['close'])

            if direction_filter in ("BUY", "BOTH"):
                single_bull = (c_curr > o_curr) and ((c_curr - o_curr) >= self.impulse_atr_mult * atr)
                double_bull = (c_curr > o_prev) and ((c_curr - o_prev) >= self.impulse_atr_mult * atr) and (c_curr > c_prev)
                if single_bull or double_bull:
                    ob = self._find_ob_before_impulse(df, abs_i, 'BUY', atr, current_price)
                    if ob and not self._is_ob_invalidated(df, ob, 'BUY', atr):
                        obs.append(ob)

            if direction_filter in ("SELL", "BOTH"):
                single_bear = (c_curr < o_curr) and ((o_curr - c_curr) >= self.impulse_atr_mult * atr)
                double_bear = (c_curr < o_prev) and ((o_prev - c_curr) >= self.impulse_atr_mult * atr) and (c_curr < c_prev)
                if single_bear or double_bear:
                    ob = self._find_ob_before_impulse(df, abs_i, 'SELL', atr, current_price)
                    if ob and not self._is_ob_invalidated(df, ob, 'SELL', atr):
                        obs.append(ob)

        obs = self._deduplicate(obs, atr)

        # Filter: only NON-retested OBs (the price hasn't come back yet)
        pending_obs = []
        for ob in obs:
            if ob.is_retested:
                continue  # Skip — already retested, use `detect()` for these
            
            # Score it (without retest bonus, max ~80/100)
            self._score_ob(ob, df, atr, htf_trend)
            
            if ob.score < min_pending_score:
                continue
            
            # Calculate entry/SL/TP levels for the pending zone
            self._calc_pending_levels(ob, atr)
            
            if ob.entry and ob.sl and ob.tp and ob.rr and ob.rr >= 1.5:
                pending_obs.append(ob)

        pending_obs.sort(key=lambda x: x.score, reverse=True)
        logger.info(
            f"[OB PENDING] {symbol} {timeframe}: {len(pending_obs)} zones d'attente "
            f"sur {len(obs)} OB détectés"
        )
        return pending_obs

    def _calc_pending_levels(self, ob: 'OrderBlock', atr: float):
        """
        Calcule les niveaux Entry/SL/TP pour un OB en attente (pending order).
        
        L'entrée est positionnée au milieu de la zone OB (là où le prix
        a la plus forte probabilité de réagir).
        """
        buf = self.sl_buffer_mult * atr
        min_sl_dist = self.min_sl_atr_mult * atr

        if ob.direction == 'BUY':
            # Entry: au milieu-bas de la zone (meilleur prix d'entrée)
            entry = (ob.ob_mid + ob.ob_low) / 2
            raw_sl = ob.ob_low - buf
            sl_dist = entry - raw_sl
            if sl_dist < min_sl_dist:
                raw_sl = entry - min_sl_dist
                sl_dist = min_sl_dist
            ob.entry = round(entry, 5)
            ob.sl = round(raw_sl, 5)
            ob.tp = round(ob.entry + sl_dist * self.tp_rr_mult, 5)
        else:
            # Sell: entry au milieu-haut de la zone
            entry = (ob.ob_mid + ob.ob_high) / 2
            raw_sl = ob.ob_high + buf
            sl_dist = raw_sl - entry
            if sl_dist < min_sl_dist:
                raw_sl = entry + min_sl_dist
                sl_dist = min_sl_dist
            ob.entry = round(entry, 5)
            ob.sl = round(raw_sl, 5)
            ob.tp = round(ob.entry - sl_dist * self.tp_rr_mult, 5)

        sl_dist = abs(ob.entry - ob.sl)
        tp_dist = abs(ob.tp - ob.entry)
        ob.rr = round(tp_dist / sl_dist, 2) if sl_dist > 0 else None

