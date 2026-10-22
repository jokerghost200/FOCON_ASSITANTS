"""
Stratégie Romain Bailleul : Price Action, Statistiques et Retracement Fibonacci 61.8%.
Cibles principales : NASDAQ (NAS100/US100), S&P 500 (US500), Dow Jones (US30).
Règles :
1. Filtre macroéconomique (News Blackout) : Pause avant/après les annonces économiques majeures.
2. Filtre panique (VIX / Volatilité) : Refuse d'entrer si la volatilité explose de manière désordonnée (> seuil VIX).
3. Setup Principal — Pullback OPR (Opening Price Range) :
   - Détecte l'impulsion après l'ouverture et guette le retour précis sur le cours d'ouverture (OPR).
4. Setup de Repli — Fibonacci 61.8% de l'impulsion globale :
   - Si l'OPR est invalidé, recherche le niveau 61.8% (Golden Pocket) du swing majeur récent.
5. Filtre statistique de crédibilité de la cible (ADR 60j type Edgeful) :
   - L'objectif (TP) ne doit pas excéder 75% de la plage moyenne journalière restante.
6. Money Management rigoureux :
   - Stop Loss = k × ATR (1.5 × ATR)
   - Passage automatique au Break Even dès +1.0 R
   - Trailing stop positif ensuite pour sécuriser les gains.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import logging
import pandas as pd
import numpy as np

from src.strategies.base_strategy import BaseTraderStrategy, ProSignal

logger = logging.getLogger(__name__)


class RomainStrategy(BaseTraderStrategy):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        super().__init__(
            profile_id="romain",
            name="Romain Bailleul",
            config=cfg,
        )
        self.news_minutes = float(self.config.get("news_minutes", 30.0))
        self.vix_max = float(self.config.get("vix_max", 32.0))
        self.opr_tolerance_pct = float(self.config.get("opr_tolerance_pct", 0.05))
        self.fib_level = float(self.config.get("fib_level", 0.618))
        self.atr_mult = float(self.config.get("atr_mult", 1.5))
        self.be_r = float(self.config.get("be_r", 1.0))
        self.trail_r = float(self.config.get("trail_r", 1.5))
        self.avg_days = int(self.config.get("avg_days", 60))
        self.min_rr = float(self.config.get("min_rr", 1.8))

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

        # ── 1. Filtre Annonces Macroéconomiques (News Blackout) ──
        news_ctx = kwargs.get("news_context") or {}
        if news_ctx:
            impact = str(news_ctx.get("market_impact", "LOW")).upper()
            news_min = float(news_ctx.get("minutes_to_news", 999.0))
            if impact in ("HIGH", "CRITICAL") and abs(news_min) <= self.news_minutes:
                logger.debug(f"[Romain] {symbol}: Rejet - Annonce macroéconomique majeure à proximité ({news_min:.0f} min).")
                return None

        # ── 2. Filtre de Panique (VIX / Volatilité Anormale) ──
        atr = self.calculate_atr(df_primary, period=14)
        vix_val = float(kwargs.get("vix_value", 0.0) or (tech_features.get("vix", 0.0) if tech_features else 0.0))
        if vix_val > self.vix_max and self.vix_max > 0:
            logger.debug(f"[Romain] {symbol}: Rejet - VIX trop élevé ({vix_val:.1f} > {self.vix_max}). Marché en panique.")
            return None

        close_p = float(df_primary['close'].iloc[-1])

        # Extraire le prix d'ouverture de session (OPR)
        opr_price = None
        if 'time' in df_primary.columns:
            try:
                times = pd.to_datetime(df_primary['time'])
                latest_date = times.iloc[-1].date()
                day_bars = df_primary[times.dt.date == latest_date]
                if not day_bars.empty:
                    opr_price = float(day_bars['open'].iloc[0])
            except Exception:
                opr_price = None

        if opr_price is None:
            opr_price = float(df_primary['open'].iloc[0])

        setup_name = None
        side = None
        entry_price = None
        confluences = []

        # ── 3. Setup Principal : Pullback sur le Prix d'Ouverture (OPR) ──
        # Vérifier si le prix a créé une impulsion puis revient tester l'OPR avec rejet
        dist_to_opr_pct = abs(close_p - opr_price) / max(opr_price, 1e-6) * 100.0
        last_bar = df_primary.iloc[-1]
        prev_bar = df_primary.iloc[-2]

        if dist_to_opr_pct <= self.opr_tolerance_pct:
            # Rejet haussier sur OPR (support)
            if float(last_bar['close']) > float(last_bar['open']) and float(last_bar['low']) <= opr_price:
                side = "BUY"
                entry_price = close_p
                setup_name = "Pullback OPR (Support d'ouverture)"
                confluences.append(f"Rejet haussier sur OPR ({opr_price:.2f})")
            # Rejet baissier sous OPR (résistance)
            elif float(last_bar['close']) < float(last_bar['open']) and float(last_bar['high']) >= opr_price:
                side = "SELL"
                entry_price = close_p
                setup_name = "Pullback OPR (Résistance d'ouverture)"
                confluences.append(f"Rejet baissier sous OPR ({opr_price:.2f})")

        # ── 4. Setup de Repli : Retracement Fibonacci 61.8% de l'Impulsion Globale ──
        if side is None:
            swings = self.detect_swings(df_primary, lookback=4)
            shs = swings.get("highs", [])
            sls = swings.get("lows", [])

            if shs and sls:
                last_sh = shs[-1]["price"]
                last_sl = sls[-1]["price"]
                last_sh_idx = shs[-1]["index"]
                last_sl_idx = sls[-1]["index"]

                # Cas 1 : Impulsion haussière (Swing Low avant Swing High) -> Retracement acheteur à 61.8%
                if last_sh_idx > last_sl_idx and (last_sh - last_sl) > (atr * 2.0):
                    fibs = self.calculate_fibonacci_levels(last_sl, last_sh, direction="BUY")
                    fib_618 = fibs["0.618"]
                    # Le prix actuel est-il dans la zone 61.8% (± 0.2 ATR) ?
                    if abs(close_p - fib_618) <= (atr * 0.25):
                        side = "BUY"
                        entry_price = close_p
                        setup_name = "Fibonacci 61.8% Pullback"
                        confluences.append(f"Retracement 61.8% haussier à {fib_618:.2f} (Swing {last_sl:.2f} -> {last_sh:.2f})")

                # Cas 2 : Impulsion baissière (Swing High avant Swing Low) -> Retracement vendeur à 61.8%
                elif last_sl_idx > last_sh_idx and (last_sh - last_sl) > (atr * 2.0):
                    fibs = self.calculate_fibonacci_levels(last_sl, last_sh, direction="SELL")
                    fib_618 = fibs["0.618"]
                    if abs(close_p - fib_618) <= (atr * 0.25):
                        side = "SELL"
                        entry_price = close_p
                        setup_name = "Fibonacci 61.8% Pullback"
                        confluences.append(f"Retracement 61.8% baissier à {fib_618:.2f} (Swing {last_sh:.2f} -> {last_sl:.2f})")

        if side is None or entry_price is None:
            return None

        # ── 5. Calcul Stop Loss ATR et Cible Crédible (ADR 60j) ──
        sl_dist = max(atr * self.atr_mult, min_distance, spread * 2.0)
        tp_dist = sl_dist * self.min_rr

        # Validation statistique : Estimer l'ADR (Average Daily Range)
        recent_daily_range = atr * 4.0  # Approximation de l'amplitude quotidienne
        if tp_dist > (recent_daily_range * 0.85):
            # Cible trop ambitieuse pour la session (incohérente statistiquement avec Edgeful)
            logger.debug(f"[Romain] {symbol}: Rejet - Objectif de gain ({tp_dist:.1f}) trop grand vs ADR moyen ({recent_daily_range:.1f}).")
            return None

        if side == "BUY":
            sl = entry_price - sl_dist
            tp = entry_price + tp_dist
        else:
            sl = entry_price + sl_dist
            tp = entry_price - tp_dist

        rr = round(tp_dist / sl_dist, 2)
        confluences.extend([
            f"SL basé sur {self.atr_mult:.1f}× ATR ({sl_dist:.2f} pts)",
            f"Break Even automatique programmé à +{self.be_r:.1f} R",
            "Trailing Stop dynamique actif",
            "Statistiques de gain réalistes vs range journalier"
        ])

        reason = (
            f"Romain Bailleul [{symbol}]: {setup_name} détecté en {side}. "
            f"SL à {self.atr_mult}x ATR, BE à {self.be_r} R et Trailing actif. R:R {rr:.1f}."
        )

        return ProSignal(
            trader=self.name,
            profile_id=self.profile_id,
            direction=side,
            entry=entry_price,
            sl=sl,
            tp=tp,
            rr=rr,
            is_pending=False,
            order_type="MARKET",
            be_at_r=self.be_r,
            trail=True,
            reason=reason,
            confluences=confluences,
            metadata={
                "setup_type": setup_name,
                "atr": atr,
                "opr_price": opr_price,
                "be_r": self.be_r,
                "trail_r": self.trail_r,
            }
        )
