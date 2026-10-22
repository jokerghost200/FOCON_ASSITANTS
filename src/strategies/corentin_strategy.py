"""
Stratégie Corentin Trading : Order Flow, Inducement et Imbalances M15.
Cible principale : EURUSD (analyse structure + carnet, exécution précise).
Règles :
1. Fenêtres horaires autorisées (ex: reversal 16h00 ± 30 min, session Londres 10h00 ± 30 min).
2. Filtre anti-range : Si l'amplitude récente / ATR < seuil (ex: 0.8), le marché manque d'impulsion -> aucun trade.
3. Biais Daily Cycle : Alignement par rapport au prix d'ouverture du jour ou EMA 50 HTF.
4. Zone d'Imbalance M15 : Fair Value Gap récent aligné avec le biais, le prix doit être à l'intérieur ou retester la zone.
5. Confirmation LTF (M5/M3) : Changement de structure (MSS) ou rejet net dans le sens du biais.
6. Filtre Order Flow optionnel : Vérifie l'absence de pression vendeuse/acheteuse violente contraire dans le carnet/CVD.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import logging
import pandas as pd
import numpy as np

from src.strategies.base_strategy import BaseTraderStrategy, ProSignal

logger = logging.getLogger(__name__)


class CorentinStrategy(BaseTraderStrategy):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        super().__init__(
            profile_id="corentin",
            name="Corentin Trading",
            config=cfg,
        )
        self.allowed_windows = self.config.get("allowed_windows", [
            {"start": "15:30", "end": "16:45"},  # Fenêtre US reversal 16h00
            {"start": "09:30", "end": "10:45"},  # Fenêtre Londres reversal 10h00
        ])
        self.range_atr_ratio = float(self.config.get("range_atr_ratio", 0.8))
        self.use_flow = bool(self.config.get("use_flow", True))
        self.min_flow_score = float(self.config.get("min_flow_score", 50.0))
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

        if df_primary is None or len(df_primary) < 20:
            return None

        now = current_dt or datetime.now()

        # ── 1. Vérification de la Fenêtre Horaire ──
        in_window = False
        active_window_desc = ""
        for win in self.allowed_windows:
            w_start = win.get("start", "15:45")
            w_end = win.get("end", "16:30")
            if self.in_time_window(now, w_start, w_end):
                in_window = True
                active_window_desc = f"{w_start}-{w_end}"
                break

        if not in_window:
            logger.debug(f"[Corentin] {symbol}: Hors des fenêtres d'exécution autorisées ({now.strftime('%H:%M')}).")
            return None

        atr = self.calculate_atr(df_primary, period=14)
        close_p = float(df_primary['close'].iloc[-1])

        # ── 2. Filtre Anti-Range (Discipline : pas de trade sans impulsion) ──
        recent_bars = df_primary.tail(10)
        recent_range = float(recent_bars['high'].max() - recent_bars['low'].min())
        range_ratio = recent_range / max(atr, 1e-6)
        if range_ratio < self.range_atr_ratio:
            logger.debug(f"[Corentin] {symbol}: Marché en range sans impulsion (Ratio {range_ratio:.2f} < {self.range_atr_ratio}).")
            return None

        # ── 3. Biais Daily Cycle ──
        # Déterminé par la position du prix par rapport à l'ouverture du jour ou tendance HTF
        bias = None
        open_day = None
        if 'time' in df_primary.columns:
            try:
                times = pd.to_datetime(df_primary['time'])
                latest_date = times.iloc[-1].date()
                day_bars = df_primary[times.dt.date == latest_date]
                if not day_bars.empty:
                    open_day = float(day_bars['open'].iloc[0])
            except Exception:
                open_day = None

        if open_day is not None:
            bias = "BUY" if close_p >= open_day else "SELL"
        else:
            trend_str = str(tech_features.get('trend', 'NEUTRAL') if tech_features else 'NEUTRAL').upper()
            bias = "BUY" if "BULL" in trend_str else "SELL" if "BEAR" in trend_str else None

        if not bias:
            return None

        # ── 4. Imbalance M15 (Fair Value Gap) ──
        # Identifier le nearest imbalance dans le sens du biais
        imbalances = self.detect_imbalances_fvg(df_primary, lookback=25)
        aligned_imbalances = [imb for imb in imbalances if imb['direction'] == bias]

        if not aligned_imbalances:
            logger.debug(f"[Corentin] {symbol}: Aucune imbalance M15 alignée avec le biais {bias}.")
            return None

        # Trouver la zone dans laquelle se trouve le prix actuel ou la plus proche (dans une marge de 0.5 ATR)
        target_fvg = None
        for fvg in reversed(aligned_imbalances):
            # Le prix est-il à l'intérieur de la zone FVG ou en train de la re-tester ?
            fvg_top = fvg['top']
            fvg_bottom = fvg['bottom']
            if fvg_bottom - (atr * 0.5) <= close_p <= fvg_top + (atr * 0.5):
                target_fvg = fvg
                break

        if not target_fvg:
            logger.debug(f"[Corentin] {symbol}: Prix hors des zones d'imbalance M15.")
            return None

        # ── 5. Confirmation LTF (M5 ou M3) ──
        # Vérifie un micro structure shift (MSS) ou une mèche de rejet confirmée
        has_ltf_confirmation = False
        confluences = [
            f"Fenêtre {active_window_desc}",
            f"Daily Bias {bias}",
            f"Imbalance M15 [{target_fvg['bottom']:.5f} - {target_fvg['top']:.5f}]"
        ]

        # Vérifier df_ltf si fourni, sinon analyser les 5 dernières barres de df_primary
        eval_df = df_ltf if df_ltf is not None and len(df_ltf) >= 5 else df_primary.tail(8)
        last_candle = eval_df.iloc[-1]
        candle_open = float(last_candle['open'])
        candle_close = float(last_candle['close'])
        candle_high = float(last_candle['high'])
        candle_low = float(last_candle['low'])

        if bias == "BUY":
            # Rejet haussier : mèche basse significative ou clôture verte après retest du bas
            lower_wick = min(candle_open, candle_close) - candle_low
            body = abs(candle_close - candle_open)
            if candle_close > candle_open or lower_wick >= body * 0.5:
                has_ltf_confirmation = True
                confluences.append("Rejet M5/LTF sur support FVG")
        elif bias == "SELL":
            # Rejet baissier : mèche haute significative ou clôture rouge après retest du haut
            upper_wick = candle_high - max(candle_open, candle_close)
            body = abs(candle_close - candle_open)
            if candle_close < candle_open or upper_wick >= body * 0.5:
                has_ltf_confirmation = True
                confluences.append("Rejet M5/LTF sous résistance FVG")

        if not has_ltf_confirmation:
            logger.debug(f"[Corentin] {symbol}: Confirmation LTF non validée.")
            return None

        # ── 6. Filtre Order Flow (Optionnel / Confluence) ──
        of_score = 0.0
        if self.use_flow and order_flow_confluence:
            of_score = getattr(order_flow_confluence, 'confluence_score', 0.0)
            of_details = getattr(order_flow_confluence, 'details', [])
            is_confirmed = getattr(order_flow_confluence, 'is_confirmed', False)

            # Vérifier qu'il n'y a pas d'opposition radicale
            if bias == "BUY" and any("Pression Vendeuse" in d or "Absorption Baissière" in d for d in of_details):
                logger.debug(f"[Corentin] {symbol}: Rejet - Flux d'ordres opposé à l'achat.")
                return None
            if bias == "SELL" and any("Pression Acheteuse" in d or "Absorption Haussière" in d for d in of_details):
                logger.debug(f"[Corentin] {symbol}: Rejet - Flux d'ordres opposé à la vente.")
                return None

            if is_confirmed or of_score >= self.min_flow_score:
                confluences.append(f"Flux d'ordres confirmé (Score {of_score:.0f}/100)")
            else:
                confluences.append(f"Flux d'ordres neutre/atténué (Score {of_score:.0f}/100)")

        # ── 7. Construction du Signal et Niveaux (SL / TP) ──
        buffer = max(spread * 1.5, min_distance, atr * 0.15)
        if bias == "BUY":
            entry = close_p
            # Stop Loss sous la borne inférieure de la zone d'imbalance + marge
            sl = float(target_fvg['bottom']) - buffer
            risk_dist = entry - sl
            if risk_dist <= 0:
                return None
            tp = entry + (risk_dist * self.min_rr)
            rr = round((tp - entry) / risk_dist, 2)
        else:
            entry = close_p
            # Stop Loss au-dessus de la borne supérieure de l'imbalance + marge
            sl = float(target_fvg['top']) + buffer
            risk_dist = sl - entry
            if risk_dist <= 0:
                return None
            tp = entry - (risk_dist * self.min_rr)
            rr = round((entry - tp) / risk_dist, 2)

        reason = (
            f"Corentin Trading [{symbol}]: Biais {bias} dans fenêtre {active_window_desc}, "
            f"imbalance M15 mitigée avec rejet LTF. RR {rr:.1f}."
        )

        return ProSignal(
            trader=self.name,
            profile_id=self.profile_id,
            direction=bias,
            entry=entry,
            sl=sl,
            tp=tp,
            rr=rr,
            is_pending=False,
            order_type="MARKET",
            reason=reason,
            confluences=confluences,
            metadata={
                "target_fvg": target_fvg,
                "range_ratio": range_ratio,
                "of_score": of_score,
                "window": active_window_desc,
            }
        )
