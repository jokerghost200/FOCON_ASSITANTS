"""
Stratégie Benjamin Mauger : Market Making & Scalping d'ouverture (US30 / NASDAQ / Gold).
Règles :
1. Fenêtre d'ouverture stricte : N premières minutes après l'ouverture de session (ex: 15:30 US Open, 09:00 Londres).
2. Contrôle rigoureux du spread : Si spread > max_spread (points), rejet immédiat car le scalping est sensible aux frais.
3. Impulsion directionnelle d'ouverture : Détecte le sens de la cassure de la bougie d'ouverture.
4. Filtre All-Time High (Anti-ATH) : Interdit les achats si le prix est à moins de X% (ex: 0.3%) du plus haut récent.
5. Grappes d'ordres limites (Cluster) : Prépare 2 à 3 ordres limites étagés (step_pts) avec Take-Profit court (tp_pts).
6. Risque dégressif : Réduit le coefficient de risque au fil des trades exécutés dans la session.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import logging
import pandas as pd
import numpy as np

from src.strategies.base_strategy import BaseTraderStrategy, ProSignal

logger = logging.getLogger(__name__)


class BenjaminStrategy(BaseTraderStrategy):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        super().__init__(
            profile_id="benjamin",
            name="Benjamin Mauger",
            config=cfg,
        )
        self.open_window_min = float(self.config.get("open_window_min", 15.0))
        self.max_spread = float(self.config.get("max_spread", 2.5))
        self.cluster_n = int(self.config.get("cluster_n", 3))
        self.step_pts = float(self.config.get("step_pts", 5.0))
        self.tp_pts = float(self.config.get("tp_pts", 8.0))
        self.sl_pts = float(self.config.get("sl_pts", 12.0))
        self.ath_pct = float(self.config.get("ath_pct", 0.3))
        self.decay_mode = str(self.config.get("decay_mode", "trades"))
        self.decay_rate = float(self.config.get("decay_rate", 0.2))
        self.executed_trades_today = 0

    def set_executed_trades_count(self, count: int):
        """Met à jour le compteur de trades de la session pour calculer le risque dégressif."""
        self.executed_trades_today = max(0, int(count))

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

        if df_primary is None or len(df_primary) < 15:
            return None

        now = current_dt or datetime.now()

        # ── 1. Vérification de la Fenêtre d'Ouverture ──
        # Les sessions clés pour US30/NAS/Gold : 15:30 CET (US Open) et 09:00 CET (London Open)
        min_since_us = self.minutes_since_time(now, "15:30")
        min_since_lon = self.minutes_since_time(now, "09:00")

        in_open_window = False
        session_label = ""
        if 0.0 <= min_since_us <= self.open_window_min:
            in_open_window = True
            session_label = f"US Open (+{min_since_us:.0f}m)"
        elif 0.0 <= min_since_lon <= self.open_window_min:
            in_open_window = True
            session_label = f"London Open (+{min_since_lon:.0f}m)"

        if not in_open_window:
            logger.debug(f"[Benjamin] {symbol}: Hors fenêtre d'ouverture (US: {min_since_us:.1f}m, Lon: {min_since_lon:.1f}m).")
            return None

        # ── 2. Contrôle du Spread (Tolérance stricte pour scalping) ──
        # Si spread fourni en prix absolu, convertir si nécessaire
        atr = self.calculate_atr(df_primary, period=14)
        spread_val = spread if spread > 0 else (atr * 0.1)
        if spread_val > self.max_spread and self.max_spread > 0:
            logger.debug(f"[Benjamin] {symbol}: Spread trop élevé pour le scalping ({spread_val:.2f} > {self.max_spread}).")
            return None

        close_p = float(df_primary['close'].iloc[-1])

        # ── 3. Sens de l'Impulsion d'Ouverture ──
        # Analyse des 3 dernières barres pour identifier l'agressivité
        recent_bars = df_primary.tail(3)
        net_move = float(recent_bars['close'].iloc[-1] - recent_bars['open'].iloc[0])
        side = "BUY" if net_move > 0 else "SELL" if net_move < 0 else None
        if not side:
            return None

        # ── 4. Filtre All-Time High / Résistance Majeure ──
        # Si achat, vérifier qu'on n'achète pas directement sous le sommet historique
        if side == "BUY":
            highest_high = float(df_primary['high'].max())
            dist_to_ath_pct = ((highest_high - close_p) / max(highest_high, 1e-6)) * 100.0
            if self.ath_pct > 0 and dist_to_ath_pct <= self.ath_pct:
                logger.debug(f"[Benjamin] {symbol}: Rejet BUY - Prix trop proche de l'ATH ({dist_to_ath_pct:.2f}% <= {self.ath_pct}%).")
                return None

        # ── 5. Calcul du Risque Dégressif ──
        # 1er trade = 1.0x, 2e = 0.8x, 3e = 0.6x, plancher à 0.4x
        decay_mult = max(0.4, 1.0 - (self.executed_trades_today * self.decay_rate))

        # ── 6. Construction de la Grappe d'Ordres (Cluster) ──
        # Calcul des paliers étagés autour du prix d'entrée
        cluster_orders = []
        point_step = max(self.step_pts, min_distance, atr * 0.25)
        tp_dist = max(self.tp_pts, point_step * 1.5)
        sl_dist = max(self.sl_pts, point_step * 2.0)

        for i in range(self.cluster_n):
            if side == "BUY":
                # Ordres limites d'achat étagés sous le prix actuel (pullback scalp)
                c_entry = close_p - (i * point_step)
                c_sl = c_entry - sl_dist
                c_tp = c_entry + tp_dist
            else:
                # Ordres limites de vente étagés au-dessus du prix actuel
                c_entry = close_p + (i * point_step)
                c_sl = c_entry + sl_dist
                c_tp = c_entry - tp_dist

            cluster_orders.append({
                "tier": i + 1,
                "side": side,
                "entry": round(c_entry, 2),
                "sl": round(c_sl, 2),
                "tp": round(c_tp, 2),
                "order_type": "BUY_LIMIT" if side == "BUY" else "SELL_LIMIT",
            })

        main_entry = cluster_orders[0]["entry"]
        main_sl = cluster_orders[0]["sl"]
        main_tp = cluster_orders[0]["tp"]
        rr = round(abs(main_tp - main_entry) / max(abs(main_entry - main_sl), 1e-6), 2)

        confluences = [
            f"Scalp {session_label}",
            f"Impulsion {side} ({net_move:+.2f} pts)",
            f"Spread maîtrisé ({spread_val:.2f} pts)",
            f"Grappe de {self.cluster_n} ordres limites (pas: {point_step:.1f} pts)",
            f"Risque dégressif appliqué (x{decay_mult:.1f})"
        ]

        reason = (
            f"Benjamin Mauger [{symbol}]: Scalping d'ouverture {session_label}, "
            f"cluster {self.cluster_n} ordres {side} avec TP court ({tp_dist:.1f} pts). Multiplicateur risque {decay_mult:.1f}x."
        )

        return ProSignal(
            trader=self.name,
            profile_id=self.profile_id,
            direction=side,
            entry=main_entry,
            sl=main_sl,
            tp=main_tp,
            rr=rr,
            is_pending=True,
            order_type="BUY_LIMIT" if side == "BUY" else "SELL_LIMIT",
            expiration_bars=3,
            expiration_seconds=900,  # Scalp court : 15 min max
            cluster_orders=cluster_orders,
            reason=reason,
            confluences=confluences,
            decay_risk_mult=decay_mult,
            metadata={
                "cluster_orders": cluster_orders,
                "session": session_label,
                "decay_mult": decay_mult,
                "step_pts": point_step,
            }
        )
