"""
Boomerang Setup Detector
========================
Stratégie : Breakout + Retest + Rejet + Confirmation multi-filtre.

Pipeline de détection :
  1. Détection de l'impulsion (mouvement fort directionnel)
  2. Confirmation du breakout (cassure franche d'un niveau)
  3. Mesure de la force du breakout (ATR, volume tick)
  4. Détection du retest (retour sur la zone cassée)
  5. Confirmation du rejet (bougie de rejet sur la zone)
  6. Filtres : BOS/MSS, FVG, HTF trend, fractal, zigzag, RSI
  7. Calcul du score global (0–100) et Risk/Reward
  8. Décision finale (VALID / WEAK / INVALID)
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Score thresholds
# ─────────────────────────────────────────────────────────────────────────────
SCORE_VALID = 60   # score >= 60 → VALID trade
SCORE_WEAK  = 40   # 40 <= score < 60 → WEAK (afficher mais ne pas auto-trader)


@dataclass
class BoomerangResult:
    """Résultat complet d'une détection Boomerang."""

    symbol:    str
    timeframe: str
    direction: str     = 'NONE'     # 'BUY' | 'SELL' | 'NONE'
    score:     float   = 0.0        # 0–100
    decision:  str     = 'INVALID'  # 'VALID' | 'WEAK' | 'INVALID'

    # Niveaux
    breakout_level: Optional[float] = None
    entry:          Optional[float] = None
    sl:             Optional[float] = None
    tp:             Optional[float] = None
    rr:             Optional[float] = None   # Risk/Reward ratio

    # Détails des sous-scores
    impulse_ok:       bool = False
    breakout_ok:      bool = False
    retest_ok:        bool = False
    rejection_ok:     bool = False
    fractal_aligned:  bool = False
    zigzag_aligned:   bool = False
    rsi_ok:           bool = False
    atr_ok:           bool = False
    htf_aligned:      bool = False

    reasons: list = field(default_factory=list)

    def summary(self) -> str:
        ok = lambda b: '[OK]' if b else '[NO]'
        lines = [
            f"BOOMERANG {self.direction} -- {self.symbol} {self.timeframe}",
            f"Breakout zone : {self.breakout_level}",
            f"Entry         : {self.entry}",
            f"SL            : {self.sl}",
            f"TP            : {self.tp}",
            f"R:R           : {self.rr:.2f}" if self.rr else "R:R : N/A",
            f"Score         : {self.score:.0f}/100",
            f"Decision      : {self.decision}",
            "-" * 40,
            f"Impulse       : {ok(self.impulse_ok)}",
            f"Breakout      : {ok(self.breakout_ok)}",
            f"Retest        : {ok(self.retest_ok)}",
            f"Rejection     : {ok(self.rejection_ok)}",
            f"Fractal align : {ok(self.fractal_aligned)}",
            f"ZigZag align  : {ok(self.zigzag_aligned)}",
            f"RSI ok        : {ok(self.rsi_ok)}",
            f"ATR ok        : {ok(self.atr_ok)}",
            f"HTF aligned   : {ok(self.htf_aligned)}",
        ]
        if self.reasons:
            lines.append("Notes : " + " | ".join(self.reasons))
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

class BoomerangDetector:
    """
    Détecte les setups Boomerang sur un DataFrame avec indicateurs calculés.

    Parameters
    ----------
    impulse_atr_mult   : float  — L'impulsion doit faire >= N × ATR sur la fenêtre.
    breakout_confirm   : int    — Nombre de bougies de confirmation du breakout.
    retest_tolerance   : float  — Tolérance du retest en fraction d'ATR (ex: 0.5).
    rejection_wick_min : float  — Wick minimum de rejet en fraction d'ATR.
    min_rr             : float  — Ratio R:R minimum pour valider le trade.
    """

    def __init__(
        self,
        impulse_atr_mult: float   = 2.0,
        breakout_confirm: int     = 1,
        retest_tolerance: float   = 0.5,
        rejection_wick_min: float = 0.3,
        min_rr: float             = 1.5,
    ):
        self.impulse_atr_mult   = impulse_atr_mult
        self.breakout_confirm   = breakout_confirm
        self.retest_tolerance   = retest_tolerance
        self.rejection_wick_min = rejection_wick_min
        self.min_rr             = min_rr

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        htf_trend: str = 'Unknown',
        amd_context: dict = None,
    ) -> BoomerangResult:
        """
        Runs the full Boomerang detection pipeline on the given DataFrame.
        The DataFrame must have already been processed by TechnicalIndicators.add_all_indicators().

        Returns a BoomerangResult (check .decision to know if trade is VALID).
        """
        result = BoomerangResult(symbol=symbol, timeframe=timeframe)

        if df is None or len(df) < 50:
            result.reasons.append('Not enough data')
            return result

        amd = amd_context or {}

        # Work on the last 100 bars for speed
        df = df.tail(100).copy().reset_index(drop=True)

        atr = float(df['ATR_14'].iloc[-1]) if 'ATR_14' in df.columns else 0.0
        if atr <= 0:
            result.reasons.append('ATR invalid')
            return result

        close   = float(df['close'].iloc[-1])
        high    = float(df['high'].iloc[-1])
        low     = float(df['low'].iloc[-1])

        # ── Step 1 : Impulse detection ────────────────────────────────────────
        bull_impulse, bear_impulse, impulse_high, impulse_low = \
            self._detect_impulse(df, atr)
        result.impulse_ok = bull_impulse or bear_impulse

        if not result.impulse_ok:
            result.reasons.append('No strong impulse detected')
            return result

        # ── Step 2 : Breakout level & direction ───────────────────────────────
        direction, breakout_level, result.breakout_ok = \
            self._confirm_breakout(df, atr, bull_impulse, bear_impulse,
                                   impulse_high, impulse_low)

        if not result.breakout_ok or direction == 'NONE':
            result.reasons.append('Breakout not confirmed')
            return result

        result.direction     = direction
        result.breakout_level = breakout_level

        # ── Step 3 : Retest detection ──────────────────────────────────────────
        result.retest_ok = self._detect_retest(
            df, breakout_level, atr, direction)

        if not result.retest_ok:
            result.reasons.append('No retest of breakout zone')

        # ── Step 4 : Rejection candle ──────────────────────────────────────────
        result.rejection_ok = self._detect_rejection(
            df, breakout_level, atr, direction)

        if not result.rejection_ok:
            result.reasons.append('No rejection candle at retest zone')

        # ── Step 5 : Filters ──────────────────────────────────────────────────
        result.fractal_aligned = self._check_fractal(df, direction)
        result.zigzag_aligned  = self._check_zigzag(df, direction)
        result.rsi_ok          = self._check_rsi(df, direction)
        result.atr_ok          = self._check_atr(df, atr)
        result.htf_aligned     = self._check_htf(htf_trend, direction)

        # BOS/MSS from AMD context
        bos_ok = bool(amd.get('BOS_MSS'))
        if bos_ok:
            result.reasons.append('BOS/MSS confirmed ✅')

        # FVG
        fvg_ok = bool(amd.get('FVG'))
        if fvg_ok:
            result.reasons.append('FVG present ✅')

        # ── Step 6 : Score ────────────────────────────────────────────────────
        score = 0.0
        score += 15.0 if result.retest_ok     else 0.0   # Retest = critique
        score += 15.0 if result.rejection_ok  else 0.0   # Rejet = critique
        score += 10.0 if result.fractal_aligned else 0.0
        score += 10.0 if result.zigzag_aligned  else 0.0
        score += 10.0 if result.rsi_ok          else 0.0
        score += 10.0 if result.atr_ok          else 0.0
        score += 15.0 if result.htf_aligned     else 0.0
        score += 10.0 if bos_ok                 else 0.0
        score += 5.0  if fvg_ok                 else 0.0
        result.score = min(score, 100.0)

        # ── Step 7 : Entry / SL / TP / R:R ───────────────────────────────────
        result.entry, result.sl, result.tp, result.rr = \
            self._calc_levels(df, direction, breakout_level, atr)

        if result.rr is not None and result.rr < self.min_rr:
            result.reasons.append(f'R:R {result.rr:.2f} < min {self.min_rr}')
            result.score *= 0.7   # pénalise le score si R:R insuffisant

        # ── Step 8 : Decision ─────────────────────────────────────────────────
        if result.score >= SCORE_VALID and result.retest_ok and result.rejection_ok:
            result.decision = 'VALID'
        elif result.score >= SCORE_WEAK:
            result.decision = 'WEAK'
        else:
            result.decision = 'INVALID'

        logger.info(result.summary())
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _detect_impulse(self, df, atr):
        """Retourne (bull_impulse, bear_impulse, impulse_high, impulse_low)."""
        window = min(10, len(df) - 1)
        recent = df.tail(window + 1).iloc[:-1]   # fenêtre excluant la dernière bougie

        total_range = float(recent['high'].max() - recent['low'].min())
        min_impulse = self.impulse_atr_mult * atr

        impulse_high = float(recent['high'].max())
        impulse_low  = float(recent['low'].min())

        bull_impulse = (float(recent['close'].iloc[-1]) - float(recent['open'].iloc[0])) > min_impulse
        bear_impulse = (float(recent['open'].iloc[0]) - float(recent['close'].iloc[-1])) > min_impulse

        return bull_impulse, bear_impulse, impulse_high, impulse_low

    def _confirm_breakout(self, df, atr, bull_impulse, bear_impulse,
                          impulse_high, impulse_low):
        """Retourne (direction, breakout_level, confirmed)."""
        last = df.iloc[-self.breakout_confirm - 1: -1]
        current_close = float(df['close'].iloc[-1])

        if bull_impulse:
            # Breakout = la bougie précédente a clôturé au-dessus du high de l'impulsion
            if float(last['close'].max()) > impulse_high:
                return 'BUY', impulse_high, True
        if bear_impulse:
            if float(last['close'].min()) < impulse_low:
                return 'SELL', impulse_low, True

        # Fallback : utilise le dernier fractal comme niveau de breakout
        if 'last_fractal_high' in df.columns and 'last_fractal_low' in df.columns:
            frac_h = float(df['last_fractal_high'].iloc[-1] or 0)
            frac_l = float(df['last_fractal_low'].iloc[-1]  or 0)
            if bull_impulse and frac_h > 0 and current_close > frac_h:
                return 'BUY', frac_h, True
            if bear_impulse and frac_l > 0 and current_close < frac_l:
                return 'SELL', frac_l, True

        return 'NONE', 0.0, False

    def _detect_retest(self, df, level, atr, direction):
        """Vérifie si le prix est revenu vers le niveau de breakout."""
        tol = self.retest_tolerance * atr
        recent_lows  = df['low'].tail(5)
        recent_highs = df['high'].tail(5)

        if direction == 'BUY':
            # Retest = une bougie récente a touché la zone (level ± tol)
            touched = ((recent_lows <= level + tol) & (recent_highs >= level - tol)).any()
            return bool(touched)
        else:
            touched = ((recent_highs >= level - tol) & (recent_lows <= level + tol)).any()
            return bool(touched)

    def _detect_rejection(self, df, level, atr, direction):
        """Détecte une bougie de rejet (pin bar / engulfing) sur la zone."""
        min_wick = self.rejection_wick_min * atr
        last = df.iloc[-1]
        o = float(last['open']); c = float(last['close'])
        h = float(last['high']); l = float(last['low'])
        body = abs(c - o)

        if direction == 'BUY':
            lower_wick = min(o, c) - l
            return lower_wick >= min_wick and c > o   # wick bas + bougie haussière
        else:
            upper_wick = h - max(o, c)
            return upper_wick >= min_wick and c < o   # wick haut + bougie baissière

    def _check_fractal(self, df, direction):
        if 'fractal_up' not in df.columns:
            return False
        last = df.iloc[-1]
        if direction == 'BUY':
            # Un fractal bas récent = support fractal = aligné
            return bool(df['fractal_down'].tail(5).any())
        else:
            return bool(df['fractal_up'].tail(5).any())

    def _check_zigzag(self, df, direction):
        if 'zigzag_trend' not in df.columns:
            return False
        trend = int(df['zigzag_trend'].iloc[-1])
        if direction == 'BUY':
            return trend > 0    # zigzag trending up
        else:
            return trend < 0

    def _check_rsi(self, df, direction):
        if 'RSI_14' not in df.columns:
            return False
        rsi = float(df['RSI_14'].iloc[-1])
        if direction == 'BUY':
            return 35 <= rsi <= 65   # ni suracheté, ni en crash
        else:
            return 35 <= rsi <= 65

    def _check_atr(self, df, atr):
        """ATR courant doit être dans la plage 0.5x–3x de l'ATR moyen."""
        if 'ATR_14' not in df.columns:
            return False
        atr_mean = float(df['ATR_14'].mean())
        if atr_mean <= 0:
            return False
        ratio = atr / atr_mean
        return 0.5 <= ratio <= 3.0

    def _check_htf(self, htf_trend: str, direction: str) -> bool:
        t = (htf_trend or '').upper()
        if direction == 'BUY'  and 'BEAR' in t: return False
        if direction == 'SELL' and 'BULL' in t: return False
        return True

    def _calc_levels(self, df, direction, breakout_level, atr):
        """Calcule Entry / SL / TP / R:R."""
        close = float(df['close'].iloc[-1])
        sl_dist_mult = 1.5   # SL = 1.5 × ATR derrière la zone
        tp_dist_mult = 3.0   # TP = 3.0 × ATR = R:R de 2.0 minimum

        if direction == 'BUY':
            entry = close
            sl    = breakout_level - sl_dist_mult * atr
            tp    = close + tp_dist_mult * atr
        elif direction == 'SELL':
            entry = close
            sl    = breakout_level + sl_dist_mult * atr
            tp    = close - tp_dist_mult * atr
        else:
            return None, None, None, None

        sl_dist = abs(entry - sl)
        tp_dist = abs(tp - entry)
        rr = (tp_dist / sl_dist) if sl_dist > 0 else None

        return (
            round(entry, 5),
            round(sl,    5),
            round(tp,    5),
            round(rr, 2) if rr else None,
        )
