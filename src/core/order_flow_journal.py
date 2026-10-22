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
        self.outcomes_file = os.path.join(log_dir, "order_flow_outcomes.jsonl")
        self._in_memory_records: List[Dict[str, Any]] = []
        os.makedirs(log_dir, exist_ok=True)
        self._load_persisted_records()

    def _load_persisted_records(self) -> None:
        """
        Recharge les signaux et les dénouements de trades persistés depuis le disque.
        Garantit la survie des statistiques et de l'analyse A/B après un redémarrage du bot.
        """
        records_by_ticket: Dict[int, Dict[str, Any]] = {}
        ordered_records: List[Dict[str, Any]] = []

        # 1. Charger les signaux
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            ordered_records.append(rec)
                            ticket = rec.get("ticket")
                            if ticket is not None:
                                records_by_ticket[int(ticket)] = rec
                        except Exception:
                            continue
            except Exception as e:
                logger.debug(f"[OF Journal] Erreur lecture signaux: {e}")

        # 2. Rejouer les dénouements réels de trades
        if os.path.exists(self.outcomes_file):
            try:
                with open(self.outcomes_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            out = json.loads(line)
                            ticket = out.get("ticket")
                            if ticket is not None and int(ticket) in records_by_ticket:
                                r = records_by_ticket[int(ticket)]
                                r["pnl_usd"] = out.get("pnl_usd")
                                r["outcome"] = out.get("outcome")
                                r["exit_reason"] = out.get("exit_reason", "")
                                r["rr_achieved"] = out.get("rr_achieved", 0.0)
                            else:
                                # Outcome sans signal initial trouvé (ex: trade antérieur)
                                fallback_rec = {
                                    "timestamp": out.get("timestamp", ""),
                                    "ticket": ticket,
                                    "symbol": "UNKNOWN",
                                    "direction": "UNKNOWN",
                                    "entry_price": 0.0,
                                    "strategy": "EXTERNAL_OUTCOME",
                                    "status": "CLOSED",
                                    "notes": "Persisted outcome rehydrated",
                                    "pnl_usd": out.get("pnl_usd"),
                                    "outcome": out.get("outcome"),
                                    "exit_reason": out.get("exit_reason", ""),
                                    "rr_achieved": out.get("rr_achieved", 0.0),
                                    "order_flow": {
                                        "confirmed": False,
                                        "score": 0.0,
                                        "data_quality": "UNKNOWN",
                                    }
                                }
                                ordered_records.append(fallback_rec)
                                if ticket is not None:
                                    records_by_ticket[int(ticket)] = fallback_rec
                        except Exception:
                            continue
            except Exception as e:
                logger.debug(f"[OF Journal] Erreur lecture outcomes: {e}")

        self._in_memory_records = ordered_records

    def log_signal(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        strategy_name: str,
        of_confluence: Any,
        status: str = "PROPOSAL",
        notes: str = "",
        ticket: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Enregistre un événement de signal avec son empreinte microstructurale Order Flow.
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        
        record = {
            "timestamp": now_utc,
            "ticket": ticket,
            "symbol": str(symbol).upper(),
            "direction": str(direction).upper(),
            "entry_price": float(entry_price),
            "strategy": str(strategy_name),
            "status": str(status),
            "notes": str(notes),
            "pnl_usd": None,
            "outcome": "OPEN",
            "exit_reason": "",
            "order_flow": {
                "confirmed": bool(getattr(of_confluence, 'is_confirmed', False)),
                "score": float(getattr(of_confluence, 'confluence_score', 0.0)),
                "data_quality": getattr(of_confluence, 'data_quality', 'ESTIMATED_QUOTE_RULE'),
                "is_dom_available": bool(getattr(of_confluence, 'is_dom_available', False)),
                "is_real_volume": bool(getattr(of_confluence, 'is_real_volume', False)),
                "is_synthetic": bool(getattr(of_confluence, 'is_synthetic', False)),
                "damping_applied": float(getattr(of_confluence, 'damping_applied', 1.0)),
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

    def log_trade_outcome(
        self,
        ticket: int,
        pnl_usd: float,
        exit_reason: str = "TP",
        rr_achieved: float = 0.0,
    ) -> bool:
        """
        Enregistre le dénouement réel d'un trade (PnL, R:R et cause de fermeture).
        Persiste immédiatement le résultat sur disque dans order_flow_outcomes.jsonl
        pour permettre la persistance et les analyses A/B à long terme.
        """
        outcome_str = "WIN" if pnl_usd > 0 else "LOSS" if pnl_usd < 0 else "BREAK_EVEN"
        updated = False
        target_record = None

        for r in reversed(self._in_memory_records):
            if r.get("ticket") == ticket:
                r["pnl_usd"] = round(float(pnl_usd), 2)
                r["outcome"] = outcome_str
                r["exit_reason"] = str(exit_reason)
                r["rr_achieved"] = round(float(rr_achieved), 2)
                updated = True
                target_record = r
                break

        # Si le ticket n'était pas dans la mémoire, on crée un enregistrement pour garder la trace
        if not updated:
            new_r = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticket": ticket,
                "symbol": "UNKNOWN",
                "direction": "UNKNOWN",
                "entry_price": 0.0,
                "strategy": "EXTERNAL_OUTCOME",
                "status": "CLOSED",
                "notes": f"Exit: {exit_reason}",
                "pnl_usd": round(float(pnl_usd), 2),
                "outcome": outcome_str,
                "exit_reason": str(exit_reason),
                "rr_achieved": round(float(rr_achieved), 2),
                "order_flow": {
                    "confirmed": False,
                    "score": 0.0,
                    "data_quality": "UNKNOWN",
                }
            }
            self._in_memory_records.append(new_r)
            updated = True

        # Persistance disque JSONL de l'événement outcome
        outcome_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticket": ticket,
            "pnl_usd": round(float(pnl_usd), 2),
            "outcome": outcome_str,
            "exit_reason": str(exit_reason),
            "rr_achieved": round(float(rr_achieved), 2),
        }
        try:
            with open(self.outcomes_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(outcome_event, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"[OF Journal] Échec écriture outcome persistant dans {self.outcomes_file}: {e}")

        return updated

    def compare_ab_performance(self) -> Dict[str, Any]:
        """
        Compare objectivement la performance A/B :
          - Groupe A : Setups CRT Pure (sans confirmation Order Flow, score < seuil)
          - Groupe B : Setups CRT + Order Flow (avec confirmation Order Flow, score >= seuil)
        Mesure le taux de réussite, le profit factor, l'espérance mathématique et le drawdown.
        """
        completed = [r for r in self._in_memory_records if r.get("outcome") in ("WIN", "LOSS", "BREAK_EVEN")]

        def _calc_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not records:
                return {
                    "count": 0, "trades": 0, "wins": 0, "losses": 0,
                    "win_rate": 0.0, "win_rate_pct": 0.0, "total_pnl": 0.0,
                    "total_pnl_usd": 0.0, "profit_factor": 0.0, "expectancy": 0.0,
                    "max_drawdown": 0.0,
                }

            pnls = [float(r.get("pnl_usd", 0.0) or 0.0) for r in records]
            wins = sum(1 for p in pnls if p > 0)
            losses = sum(1 for p in pnls if p < 0)
            gross_win = sum(p for p in pnls if p > 0)
            gross_loss = abs(sum(p for p in pnls if p < 0))

            pf = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
            win_rate = (wins / len(pnls)) * 100.0 if pnls else 0.0
            tot_pnl = sum(pnls)
            expectancy = tot_pnl / len(pnls) if pnls else 0.0

            # Calcul du Max Drawdown
            equity = np.cumsum(pnls)
            peak = np.maximum.accumulate(equity)
            dd = peak - equity
            max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

            return {
                "count": len(records),
                "trades": len(records),
                "wins": wins,
                "losses": losses,
                "win_rate": round(win_rate, 1),
                "win_rate_pct": round(win_rate, 1),
                "total_pnl": round(tot_pnl, 2),
                "total_pnl_usd": round(tot_pnl, 2),
                "profit_factor": round(float(pf), 2),
                "expectancy": round(float(expectancy), 2),
                "max_drawdown": round(max_dd, 2),
            }

        group_a = [r for r in completed if not r.get("order_flow", {}).get("confirmed")]
        group_b = [r for r in completed if r.get("order_flow", {}).get("confirmed")]

        metrics_a = _calc_metrics(group_a)
        metrics_b = _calc_metrics(group_b)

        wr_delta = round(metrics_b["win_rate"] - metrics_a["win_rate"], 1)
        pf_delta = round(metrics_b["profit_factor"] - metrics_a["profit_factor"], 2)
        pnl_delta = round(metrics_b["total_pnl"] - metrics_a["total_pnl"], 2)

        md_report = (
            f"=== 📊 A/B Testing Quantitatif : CRT Pur vs CRT + Order Flow ===\n"
            f"• Trades Totaux Évalués : {len(completed)}\n\n"
            f"| Métrique             | Groupe A (CRT Pur) | Groupe B (CRT + Order Flow) | Écart / Delta |\n"
            f"|----------------------|--------------------|-----------------------------|---------------|\n"
            f"| Trades Réalisés      | {metrics_a['trades']:<18} | {metrics_b['trades']:<27} |               |\n"
            f"| Taux de Victoire (%) | {metrics_a['win_rate']:<17.1f}% | {metrics_b['win_rate']:<26.1f}% | {wr_delta:+5.1f}%       |\n"
            f"| Profit Factor        | {metrics_a['profit_factor']:<18.2f} | {metrics_b['profit_factor']:<27.2f} | {pf_delta:+5.2f}        |\n"
            f"| PnL Total ($)        | ${metrics_a['total_pnl']:<17.2f} | ${metrics_b['total_pnl']:<26.2f} | ${pnl_delta:+6.2f}       |\n"
            f"| Espérance / Trade    | ${metrics_a['expectancy']:<17.2f} | ${metrics_b['expectancy']:<26.2f} |               |\n"
            f"| Max Drawdown ($)     | ${metrics_a['max_drawdown']:<17.2f} | ${metrics_b['max_drawdown']:<26.2f} |               |\n\n"
            f"Conclusion : {'✅ Le filtre Order Flow améliore la performance globale.' if wr_delta > 0 and pf_delta >= 0 else '⚠️ Données en cours de convergence ou pas de surperformance nette.'}"
        )

        delta_dict = {
            "win_rate_diff_pct": wr_delta,
            "profit_factor_diff": pf_delta,
            "pnl_diff_usd": pnl_delta,
            "pnl_difference_usd": pnl_delta,
            "order_flow_improves_performance": bool(wr_delta > 0 and pf_delta >= 0),
        }

        return {
            "status": "SUCCESS",
            "total_evaluated_trades": len(completed),
            "group_a_crt_pure": metrics_a,
            "group_b_crt_order_flow": metrics_b,
            "delta": delta_dict,
            "delta_outperformance": delta_dict,
            "markdown_report": md_report,
        }

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
        Permet le test algorithmique du moteur Order Flow (CVD, Absorption, Footprint, Volume Profile).
        ATTENTION : Les ticks synthétiques ne constituent pas un flux réel et ne doivent pas
        servir seuls de preuve de rentabilité en trading réel.
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

        res_df = pd.DataFrame(rows)
        # Marquage explicite pour éviter toute confusion avec des ticks réels
        res_df.attrs["is_synthetic"] = True
        res_df.attrs["data_quality"] = "SYNTHETIC_SIMULATION"
        res_df.attrs["has_real_volume"] = False
        return res_df
