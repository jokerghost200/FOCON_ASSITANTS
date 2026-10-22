"""
Order Flow Journal & Backtest Metrics
======================================
Module 5 : Enregistrement, traçabilité et métriques d'évaluation
pour les signaux confirmés par l'Order Flow (DOM, CVD, Volume Profile, Absorption).
Permet de mesurer objectivement l'impact de l'Order Flow sur les performances du bot.
"""

import logging
import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OrderFlowJournal:
    """
    Journal dédié à l'enregistrement et à l'analyse rétrospective des confluences Order Flow.
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "order_flow_signals.jsonl")
        self._in_memory_records: List[Dict[str, Any]] = []
        os.makedirs(log_dir, exist_ok=True)

    def log_signal(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        strategy_name: str,
        of_confluence: Any,
        status: str = "PROPOSAL",
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Enregistre un événement de signal avec son empreinte microstructurale Order Flow.
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        
        record = {
            "timestamp": now_utc,
            "symbol": str(symbol).upper(),
            "direction": str(direction).upper(),
            "entry_price": float(entry_price),
            "strategy": str(strategy_name),
            "status": str(status),
            "notes": str(notes),
            "order_flow": {
                "confirmed": bool(getattr(of_confluence, 'is_confirmed', False)),
                "score": float(getattr(of_confluence, 'confluence_score', 0.0)),
                "cvd_confirmed": bool(getattr(of_confluence, 'cvd_confirmed', False)),
                "absorption_confirmed": bool(getattr(of_confluence, 'absorption_confirmed', False)),
                "dom_confirmed": bool(getattr(of_confluence, 'dom_confirmed', False)),
                "vp_confirmed": bool(getattr(of_confluence, 'volume_profile_confirmed', False)),
                "footprint_confirmed": bool(getattr(of_confluence, 'footprint_confirmed', False)),
                "poc": getattr(of_confluence, 'poc', None),
                "vah": getattr(of_confluence, 'vah', None),
                "val": getattr(of_confluence, 'val', None),
                "details": getattr(of_confluence, 'details', []),
            }
        }

        self._in_memory_records.append(record)

        # Persistence JSONL
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"[OF Journal] Échec écriture fichier journal: {e}")

        return record

    def get_summary(self) -> Dict[str, Any]:
        """
        Génère les statistiques synthétiques des signaux enregistrés.
        """
        total = len(self._in_memory_records)
        if total == 0:
            return {
                "total_signals": 0,
                "confirmed_count": 0,
                "confirmed_pct": 0.0,
                "avg_score": 0.0,
            }

        confirmed_count = sum(1 for r in self._in_memory_records if r.get("order_flow", {}).get("confirmed"))
        scores = [r.get("order_flow", {}).get("score", 0.0) for r in self._in_memory_records]

        return {
            "total_signals": total,
            "confirmed_count": confirmed_count,
            "confirmed_pct": round((confirmed_count / total) * 100.0, 1),
            "avg_score": round(float(np.mean(scores)), 1),
            "min_score": round(float(np.min(scores)), 1),
            "max_score": round(float(np.max(scores)), 1),
        }

    @staticmethod
    def generate_synthetic_ticks_from_candles(
        candles_df: pd.DataFrame,
        ticks_per_bar: int = 15,
        spread: float = 0.00010,
    ) -> pd.DataFrame:
        """
        Générateur de ticks synthétiques à microstructure réaliste à partir de bougies OHLCV.
        Permet le backtest précis du moteur Order Flow (CVD, Absorption, Footprint, Volume Profile)
        même en l'absence de flux de ticks réels archivés.
        """
        if candles_df is None or candles_df.empty:
            return pd.DataFrame()

        rows = []
        for dt, candle in candles_df.iterrows():
            o = float(candle['open'])
            h = float(candle['high'])
            l = float(candle['low'])
            c = float(candle['close'])
            tot_v = float(candle.get('volume', candle.get('tick_volume', ticks_per_bar)))
            v_per_tick = max(1.0, tot_v / ticks_per_bar)

            # Heuristique du profil intra-barre : O -> H -> L -> C si bear, ou O -> L -> H -> C si bull
            is_bull = c >= o
            if is_bull:
                path = [o, l, (o + h) / 2.0, h, c]
            else:
                path = [o, h, (o + l) / 2.0, l, c]

            sub_prices = np.interp(
                np.linspace(0, 1, ticks_per_bar),
                np.linspace(0, 1, len(path)),
                path
            )

            prev_p = o
            for p in sub_prices:
                # Bruit réaliste
                p_noisy = float(p + np.random.normal(0, max(1e-5, (h - l) * 0.03)))
                bid = p_noisy - (spread / 2.0)
                ask = p_noisy + (spread / 2.0)
                side = "BUY" if p_noisy >= prev_p else "SELL"
                delta = v_per_tick if side == "BUY" else -v_per_tick

                rows.append({
                    "price": round(p_noisy, 5),
                    "bid": round(bid, 5),
                    "ask": round(ask, 5),
                    "vol": round(v_per_tick, 2),
                    "side": side,
                    "delta": round(delta, 2),
                })
                prev_p = p_noisy

        return pd.DataFrame(rows)
