"""
Stratégie Nada FX : BMC (Breakout, Market Structure, Confirmation) & Fibonacci 50-61.8%.
Cibles principales : Gold (XAUUSD / GOLD), NASDAQ (NAS100 / US100).
Règles :
1. Range de référence de Londres : Calcul du plus haut (London High) et du plus bas (London Low) entre 08:00 et 15:00 CET.
2. Manipulation à l'ouverture US (Sweep de Londres) :
   - Mèche qui perce le London High ou London Low autour de l'ouverture américaine (15:15 - 16:45 CET),
     puis réintégration dans le range (reversal side).
3. Cassure de structure H1 (BOS / CHoCH) :
   - La réaction post-manipulation doit casser la structure locale dans le sens du retournement.
4. Entrée sur Retest dans la zone Fibonacci 50% - 61.8% (Golden Pocket) :
   - Ordre limite positionné dans la zone de retest de la jambe ayant causé la cassure.
   - Ne jamais entrer en FOMO sur la cassure directe !
5. Stop Loss placé au-delà du swing d'invalidation (la mèche de manipulation de Londres).
6. Take Profit avec un ratio Risk/Reward minimum de 2.0.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import logging
import pandas as pd
import numpy as np

from src.strategies.base_strategy import BaseTraderStrategy, ProSignal

logger = logging.getLogger(__name__)


class NadaStrategy(BaseTraderStrategy):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        super().__init__(
            profile_id="nada",
            name="Nada FX",
            config=cfg,
        )
        self.london_session = self.config.get("london_session", {"start": "08:00", "end": "15:00"})
        self.us_open_window = self.config.get("us_open_window", {"start": "15:15", "end": "16:45"})
        self.fib_zone_min = float(self.config.get("fib_zone_min", 0.50))
        self.fib_zone_max = float(self.config.get("fib_zone_max", 0.618))
        self.min_rr = float(self.config.get("min_rr", 2.0))

    def evaluate(
        self,
        symbol: str,
        df_primary: pd.DataFrame,
        df_htf: Optional[pd.DataFrame] = None,
        df_ltf: Optional[pd.DataFrame] = None,
        spread: float = 0.0,
        min_distance: float = 0.0,
        tech_features: Optional[Dict[str, Any]] = None,
        amd_context: Optional[Dict[str, Any]] = None,
        order_flow_confluence: Optional[Any] = None,
        current_dt: Optional[datetime] = None,
        **kwargs: Any,
    ) -> Optional[ProSignal]:
        if not self.is_symbol_supported(symbol):
            return None

        if df_primary is None or len(df_primary) < 30:
            return None

        now = current_dt or datetime.now()
        atr = self.calculate_atr(df_primary, period=14)

        # ── 1. Range de Référence de la Session de Londres (08h00 - 15h00) ──
        london_range = self.detect_session_range(df_primary, start_hour=8, end_hour=15)
        if not london_range:
            logger.debug(f"[Nada FX] {symbol}: Données insuffisantes pour établir le range de Londres.")
            return None

        lon_high = london_range["high"]
        lon_low = london_range["low"]
        close_p = float(df_primary['close'].iloc[-1])

        # ── 2. Détection de Manipulation / Sweep à l'Ouverture US ──
        # Vérifier si l'heure actuelle est dans ou proche de la fenêtre US Open
        w_start = self.us_open_window.get("start", "15:15")
        w_end = self.us_open_window.get("end", "16:45")
        in_us_window = self.in_time_window(now, w_start, w_end)

        # Scanner les dernières barres pour détecter le sweep des bornes de Londres
        recent_bars = df_primary.tail(15)
        sweep_detected = None
        sweep_extreme = None
        reversal_side = None

        low_idx = recent_bars['low'].idxmin()
        high_idx = recent_bars['high'].idxmax()
        lowest_val = float(recent_bars.loc[low_idx, 'low'])
        highest_val = float(recent_bars.loc[high_idx, 'high'])

        is_low_sweep = (lowest_val < lon_low and close_p > lon_low)
        is_high_sweep = (highest_val > lon_high and close_p < lon_high)

        if is_low_sweep and is_high_sweep:
            # Si les deux extrêmes ont été dépassés dans la fenêtre :
            # L'extrême touché en premier constitue la manipulation initiale (sweep) qui a lancé l'impulsion.
            if low_idx < high_idx:
                sweep_detected = "BULLISH_SWEEP_LONDON_LOW"
                sweep_extreme = lowest_val
                reversal_side = "BUY"
            else:
                sweep_detected = "BEARISH_SWEEP_LONDON_HIGH"
                sweep_extreme = highest_val
                reversal_side = "SELL"
        elif is_low_sweep:
            sweep_detected = "BULLISH_SWEEP_LONDON_LOW"
            sweep_extreme = lowest_val
            reversal_side = "BUY"
        elif is_high_sweep:
            sweep_detected = "BEARISH_SWEEP_LONDON_HIGH"
            sweep_extreme = highest_val
            reversal_side = "SELL"

        # Si aucun sweep récent des extrêmes de Londres n'est trouvé
        if not sweep_detected or reversal_side is None:
            # Fallback : vérifier via amd_context si un Liquidity Grab est signalé
            if amd_context:
                l_grab = str(amd_context.get('Liquidity_Grab', 'None'))
                if "Sweep Low" in l_grab:
                    sweep_detected = "BULLISH_SWEEP"
                    sweep_extreme = float(recent_bars['low'].min())
                    reversal_side = "BUY"
                elif "Sweep High" in l_grab:
                    sweep_detected = "BEARISH_SWEEP"
                    sweep_extreme = float(recent_bars['high'].max())
                    reversal_side = "SELL"

        if not sweep_detected or reversal_side is None or sweep_extreme is None:
            logger.debug(f"[Nada FX] {symbol}: Aucune manipulation / sweep de Londres détecté.")
            return None

        # ── 3. Cassure de Structure H1 / M15 (BOS ou CHoCH) ──
        # Déterminer la jambe d'impulsion post-sweep
        swings = self.detect_swings(df_primary, lookback=3)
        shs = swings.get("highs", [])
        sls = swings.get("lows", [])

        leg_start = sweep_extreme
        leg_end = close_p
        has_structure_break = False

        if reversal_side == "BUY":
            # Pour un achat : on cherche le sommet récent créé après le sweep
            post_sweep_highs = [h["price"] for h in shs if h["price"] > leg_start]
            leg_end = max(post_sweep_highs) if post_sweep_highs else float(recent_bars['high'].max())
            # Y a-t-il eu cassure d'un sommet précédent (BOS haussier) ?
            if leg_end - leg_start >= atr * 1.5:
                has_structure_break = True
        else:
            # Pour une vente : on cherche le creux récent créé après le sweep
            post_sweep_lows = [l["price"] for l in sls if l["price"] < leg_start]
            leg_end = min(post_sweep_lows) if post_sweep_lows else float(recent_bars['low'].min())
            if leg_start - leg_end >= atr * 1.5:
                has_structure_break = True

        if not has_structure_break:
            logger.debug(f"[Nada FX] {symbol}: Impulsion post-sweep insuffisante pour valider le BOS.")
            return None

        # ── 4. Retest dans la Zone Fibonacci 50% - 61.8% (Golden Pocket) ──
        if reversal_side == "BUY":
            fibs = self.calculate_fibonacci_levels(leg_start, leg_end, direction="BUY")
            fib_50 = fibs["0.50"]
            fib_618 = fibs["0.618"]
            zone_high = max(fib_50, fib_618)
            zone_low = min(fib_50, fib_618)
            # Entrée idéale au milieu de la zone (ou ordre limite)
            entry = (fib_50 + fib_618) / 2.0
            # Le prix actuel est-il dans ou au-dessus de la zone pour placer un pending limit ?
            is_pending = close_p > zone_high
            order_type = "BUY_LIMIT" if is_pending else "MARKET"
            if not is_pending:
                entry = close_p

            # Invalidation Stop Loss sous le sweep extreme de Londres
            buffer = max(spread * 2.0, min_distance, atr * 0.15)
            sl = leg_start - buffer
            risk_dist = entry - sl
            if risk_dist <= 0:
                return None
            tp = entry + (risk_dist * self.min_rr)
            rr = round((tp - entry) / risk_dist, 2)

        else:
            fibs = self.calculate_fibonacci_levels(leg_end, leg_start, direction="SELL")
            fib_50 = fibs["0.50"]
            fib_618 = fibs["0.618"]
            zone_high = max(fib_50, fib_618)
            zone_low = min(fib_50, fib_618)
            entry = (fib_50 + fib_618) / 2.0
            is_pending = close_p < zone_low
            order_type = "SELL_LIMIT" if is_pending else "MARKET"
            if not is_pending:
                entry = close_p

            buffer = max(spread * 2.0, min_distance, atr * 0.15)
            sl = leg_start + buffer
            risk_dist = sl - entry
            if risk_dist <= 0:
                return None
            tp = entry - (risk_dist * self.min_rr)
            rr = round((entry - tp) / risk_dist, 2)

        confluences = [
            f"Sweep de Londres ({sweep_detected})",
            f"Range Londres [{lon_low:.2f} - {lon_high:.2f}]",
            f"Cassure de structure (BOS) validée",
            f"Zone Golden Pocket Fibo 50-61.8% [{zone_low:.2f} - {zone_high:.2f}]",
            f"SL placé sur invalidation ({leg_start:.2f})"
        ]

        reason = (
            f"Nada FX [{symbol}]: Manipulation {sweep_detected} avec réintégration, "
            f"BOS confirmé, entrée sur retest Fibonacci 50-61.8% ({order_type}). R:R {rr:.1f}."
        )

        return ProSignal(
            trader=self.name,
            profile_id=self.profile_id,
            direction=reversal_side,
            entry=entry,
            sl=sl,
            tp=tp,
            rr=rr,
            is_pending=is_pending,
            order_type=order_type,
            expiration_bars=4,
            expiration_seconds=3600,
            reason=reason,
            confluences=confluences,
            metadata={
                "london_range": london_range,
                "sweep_extreme": leg_start,
                "fib_50": fib_50,
                "fib_618": fib_618,
                "zone_low": zone_low,
                "zone_high": zone_high,
            }
        )
