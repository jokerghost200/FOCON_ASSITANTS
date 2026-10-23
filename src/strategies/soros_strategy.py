"""
Stratégie George Soros : Global Macro, Réflexivité & Asymétrie Risque/Rendement.
Cibles principales : GBPUSD, EURUSD, USDJPY, Gold (XAUUSD), US500, US30.

Règles inspirées du légendaire investisseur Global Macro :
1. Analyse Macro & Détection de Distorsion Réflexive :
   - Selon la théorie de la réflexivité de Soros, les marchés s'auto-alimentent dans des boucles
     de rétroaction positives (bulles spéculatives ou paniques démesurées) jusqu'à ce que le prix
     s'éloigne considérablement de sa valeur d'équilibre fondamentale (EMA 200 institutionnelle).
   - Mesure de la distorsion : Distance à l'EMA 200 normalisée par l'ATR (z_atr >= deviation_atr_mult)
     ou état de climax de momentum (RSI en zone de sur-extension > 68 ou < 32).
2. Attente du Point de Rupture (Pas d'anticipation aveugle) :
   - Soros ne combat jamais une tendance réflexive tant que la boucle d'amplification est active.
   - Il attend le point de rupture où la réalité économique craque la structure :
     Cassure nette de structure de marché (MSS / BOS) sur le timeframe d'entrée.
3. Asymétrie Risque / Rendement & Invalidation Stricte :
   - "I'm only rich because I know when I'm wrong."
   - Le Stop Loss est placé au niveau précis de l'invalidation de la thèse (l'extrême du climax réflexif).
     Si ce sommet/creux est franchi, la thèse macro est invalidée et la perte est immédiatement coupée.
   - Le Take Profit vise un retour macro vers la juste valeur ou équilibre (EMA 200) avec un R:R minimum garanti (>= 2.5).
4. Catalyseurs Macro & Sentiment :
   - Prise en compte des annonces de banques centrales, chocs macroéconomiques et flux d'ordres institutionnels.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import logging
import pandas as pd
import numpy as np

from src.strategies.base_strategy import BaseTraderStrategy, ProSignal

logger = logging.getLogger(__name__)


class SorosStrategy(BaseTraderStrategy):
    """
    Stratégie Global Macro & Réflexivité de George Soros.
    Recherche les distorsions extrêmes de marché et entre sur cassure de structure avec haut R:R.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        super().__init__(
            profile_id="soros",
            name="George Soros",
            config=cfg,
        )
        self.ema_macro_period = int(self.config.get("ema_macro_period", 200))
        self.deviation_atr_mult = float(self.config.get("deviation_atr_mult", 2.0))
        self.rsi_period = int(self.config.get("rsi_period", 14))
        self.rsi_overbought = float(self.config.get("rsi_overbought", 68.0))
        self.rsi_oversold = float(self.config.get("rsi_oversold", 32.0))
        self.min_rr = float(self.config.get("min_rr", 2.5))
        self.risk_pct = float(self.config.get("risk_pct", 1.0))

    @staticmethod
    def calculate_ema(series: pd.Series, period: int = 200) -> pd.Series:
        """Calcule la Moyenne Mobile Exponentielle (EMA)."""
        if series is None or len(series) == 0:
            return pd.Series(dtype=float)
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> float:
        """Calcule le Relative Strength Index (RSI)."""
        if series is None or len(series) < period + 1:
            return 50.0
        try:
            delta = series.diff()
            gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
            rs = gain / loss.replace(0.0, 1e-9)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            val = float(rsi.iloc[-1])
            return val if not np.isnan(val) else 50.0
        except Exception:
            return 50.0

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

        if df_primary is None or len(df_primary) < 25:
            return None

        atr = self.calculate_atr(df_primary, period=14)
        if atr <= 0:
            atr = 0.001

        # Filtre de spread excessif
        if spread > 0 and spread > (0.6 * atr):
            logger.debug(f"[Soros] {symbol}: Rejet - Spread trop élevé ({spread:.5f} > 0.6x ATR {atr:.5f}).")
            return None

        # ── 1. ÉVALUATION DE L'ÉQUILIBRE MACROÉCONOMIQUE (EMA 200) ──
        # Utiliser le dataframe HTF si disponible (ex: H1/H4), sinon df_primary
        df_macro = df_htf if (df_htf is not None and len(df_htf) >= 30) else df_primary
        close_series = df_macro['close']
        ema_period = min(self.ema_macro_period, max(15, len(close_series) - 2))
        ema_series = self.calculate_ema(close_series, period=ema_period)
        ema_macro = float(ema_series.iloc[-1])

        close_p = float(df_primary['close'].iloc[-1])
        diff_ema = close_p - ema_macro
        z_atr = diff_ema / atr

        # Momentum RSI
        rsi_val = self.calculate_rsi(df_primary['close'], period=self.rsi_period)

        # ── 2. DÉTECTION D'UNE DISTORSION RÉFLEXIVE DU MARCHÉ ──
        # Bulle réflexive haussière (candidat SELL) : Prix largement au-dessus de l'équilibre OU RSI sur-étiré
        is_bullish_bubble = (z_atr >= self.deviation_atr_mult) or (rsi_val >= self.rsi_overbought)

        # Panique réflexive baissière (candidat BUY) : Prix largement sous l'équilibre OU RSI survendu
        is_bearish_panic = (z_atr <= -self.deviation_atr_mult) or (rsi_val <= self.rsi_oversold)

        if not is_bullish_bubble and not is_bearish_panic:
            logger.debug(
                f"[Soros] {symbol}: Marché à l'équilibre ou sans distorsion réflexive suffisante "
                f"(z_atr={z_atr:.2f}, rsi={rsi_val:.1f})."
            )
            return None

        # ── 3. DÉTECTION DU POINT DE RUPTURE (MARKET STRUCTURE SHIFT) ──
        # Examiner les 15 dernières bougies pour trouver le sommet/creux d'épuisement et le pivot intermédiaire
        recent_bars = df_primary.tail(15)
        highs = recent_bars['high'].values
        lows = recent_bars['low'].values
        closes = recent_bars['close'].values

        extreme_high = float(np.max(highs))
        extreme_low = float(np.min(lows))
        high_idx = int(np.argmax(highs))
        low_idx = int(np.argmin(lows))

        confluences = []

        # ─── SCÉNARIO A : VENTE À DÉCOUVERT (SHORT RÉFLEXIF - Style Black Wednesday 1992) ───
        if is_bullish_bubble and high_idx < len(recent_bars) - 1:
            # Le sommet d'épuisement a été formé il y a quelques barres.
            # Chercher le plus bas local de consolidation post-sommet (MSS)
            post_peak_lows = lows[high_idx:]
            if len(post_peak_lows) >= 2:
                local_support = float(np.min(post_peak_lows[:-1]))
                latest_close = closes[-1]
                latest_open = float(recent_bars['open'].iloc[-1])

                # Cassure de structure : la clôture actuelle casse sous le support local
                # OU bougie d'invalidation baissière nette (clôture sous l'ouverture précédente)
                is_breakdown = (latest_close < local_support) or (
                    latest_close < float(recent_bars['low'].iloc[-2]) and latest_close < latest_open
                )

                if is_breakdown:
                    confluences.append(f"Distorsion Réflexive Haussière (+{z_atr:.1f} ATR vs EMA200)")
                    confluences.append(f"RSI Climax ({rsi_val:.1f})")
                    confluences.append("Cassure de Structure (MSS Baissier)")

                    # Confluences Macro & Flux
                    news_ctx = kwargs.get("news_context")
                    if news_ctx and isinstance(news_ctx, dict):
                        impact = str(news_ctx.get("market_impact", "")).upper()
                        if impact in ("HIGH", "CRITICAL"):
                            confluences.append("Catalyseur Macroéconomique Actif")

                    if order_flow_confluence:
                        confluences.append("Pression vendeuse au carnet d'ordres")

                    # Invalidation stricte : au-dessus du sommet réflexif
                    sl = extreme_high + (0.25 * atr)
                    risk_dist = max(0.0001, sl - close_p)

                    # Cible de Take Profit : Rééquilibrage vers l'EMA 200 avec minimum garanti
                    tp_target = ema_macro if (close_p - ema_macro) >= (self.min_rr * risk_dist) else (close_p - self.min_rr * risk_dist)
                    actual_rr = (close_p - tp_target) / risk_dist

                    if actual_rr >= (self.min_rr - 1e-4):
                        reason = (
                            f"Épuisement de la bulle réflexive haussière sur {symbol} à {extreme_high:.5f}. "
                            f"Rupture de support à {local_support:.5f}. Vente asymétrique R:R {actual_rr:.1f} "
                            f"visant le rééquilibrage macro vers {tp_target:.5f}. Invalidation stricte à {sl:.5f}."
                        )
                        return ProSignal(
                            trader="George Soros",
                            profile_id="soros",
                            direction="SELL",
                            entry=close_p,
                            sl=sl,
                            tp=tp_target,
                            rr=round(actual_rr, 2),
                            is_pending=False,
                            order_type="MARKET",
                            reason=reason,
                            confluences=confluences,
                            metadata={
                                "z_atr": round(z_atr, 2),
                                "rsi": round(rsi_val, 1),
                                "ema_macro": round(ema_macro, 5),
                                "extreme_high": round(extreme_high, 5),
                                "style": "Global Macro & Réflexivité",
                            },
                        )

        # ─── SCÉNARIO B : ACHAT DE RETOURNEMENT (LONG CONTRE PANIQUE IRRATIONNELLE) ───
        if is_bearish_panic and low_idx < len(recent_bars) - 1:
            # Le creux de panique a été formé.
            post_trough_highs = highs[low_idx:]
            if len(post_trough_highs) >= 2:
                local_resistance = float(np.max(post_trough_highs[:-1]))
                latest_close = closes[-1]
                latest_open = float(recent_bars['open'].iloc[-1])

                is_breakout = (latest_close > local_resistance) or (
                    latest_close > float(recent_bars['high'].iloc[-2]) and latest_close > latest_open
                )

                if is_breakout:
                    confluences.append(f"Panique Réflexive Baissière ({z_atr:.1f} ATR vs EMA200)")
                    confluences.append(f"RSI Épuisement Vendeur ({rsi_val:.1f})")
                    confluences.append("Cassure de Structure (MSS Haussier)")

                    news_ctx = kwargs.get("news_context")
                    if news_ctx and isinstance(news_ctx, dict):
                        confluences.append("Catalyseur Macroéconomique")

                    if order_flow_confluence:
                        confluences.append("Absorption acheteuse détectée")

                    sl = extreme_low - (0.25 * atr)
                    risk_dist = max(0.0001, close_p - sl)

                    tp_target = ema_macro if (ema_macro - close_p) >= (self.min_rr * risk_dist) else (close_p + self.min_rr * risk_dist)
                    actual_rr = (tp_target - close_p) / risk_dist

                    if actual_rr >= (self.min_rr - 1e-4):
                        reason = (
                            f"Rejet de la panique réflexive vendeuse sur {symbol} à {extreme_low:.5f}. "
                            f"Reprise de structure au-dessus de {local_resistance:.5f}. Achat asymétrique R:R {actual_rr:.1f} "
                            f"visant le rééquilibrage macro vers {tp_target:.5f}. Invalidation stricte à {sl:.5f}."
                        )
                        return ProSignal(
                            trader="George Soros",
                            profile_id="soros",
                            direction="BUY",
                            entry=close_p,
                            sl=sl,
                            tp=tp_target,
                            rr=round(actual_rr, 2),
                            is_pending=False,
                            order_type="MARKET",
                            reason=reason,
                            confluences=confluences,
                            metadata={
                                "z_atr": round(z_atr, 2),
                                "rsi": round(rsi_val, 1),
                                "ema_macro": round(ema_macro, 5),
                                "extreme_low": round(extreme_low, 5),
                                "style": "Global Macro & Réflexivité",
                            },
                        )

        return None
