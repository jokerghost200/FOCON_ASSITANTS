"""
FATI Performance Analyzer (Analyseur Quantitatif de Performance)
================================================================
Calcule les métriques avancées d'évaluation d'un portefeuille de stratégies :
- Win Rate, Profit Factor, Expectancy en R, Drawdown Max,
- Ratios de Sharpe et Sortino (avec rendements normalisés en R),
- Découpage par Régime de marché, Session horaire et Famille de setup.
"""

from typing import List, Dict, Any, Optional
import math
import logging

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Moteur de diagnostic statistique pur.
    Fournit des métriques exploitables pour la validation hors échantillon.
    """

    @staticmethod
    def analyze_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse complète d'un échantillon de trades clôturés."""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy_r": 0.0,
                "net_pnl_usd": 0.0,
                "max_drawdown_usd": 0.0,
                "sharpe_ratio": 0.0,
                "by_strategy": {},
                "by_regime": {},
            }

        total = len(trades)
        wins = [t for t in trades if float(t.get("pnl_usd", 0.0) or 0.0) > 0.0]
        losses = [t for t in trades if float(t.get("pnl_usd", 0.0) or 0.0) < 0.0]
        be = [t for t in trades if float(t.get("pnl_usd", 0.0) or 0.0) == 0.0]

        gross_profit = sum(float(t.get("pnl_usd", 0.0) or 0.0) for t in wins)
        gross_loss = abs(sum(float(t.get("pnl_usd", 0.0) or 0.0) for t in losses))
        net_pnl = gross_profit - gross_loss

        profit_factor = round(gross_profit / max(gross_loss, 1e-5), 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        win_rate = round((len(wins) / total) * 100.0, 1)

        r_multiples = [float(t.get("result_r", 0.0) or 0.0) for t in trades]
        expectancy_r = round(sum(r_multiples) / max(total, 1), 3)

        # Calcul Drawdown
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in trades:
            cum_pnl += float(t.get("pnl_usd", 0.0) or 0.0)
            if cum_pnl > peak:
                peak = cum_pnl
            dd = peak - cum_pnl
            if dd > max_dd:
                max_dd = dd

        # Calcul du Sharpe Ratio en unités R
        sharpe = 0.0
        if len(r_multiples) > 1:
            mean_r = sum(r_multiples) / len(r_multiples)
            var_r = sum((x - mean_r) ** 2 for x in r_multiples) / (len(r_multiples) - 1)
            std_r = math.sqrt(var_r)
            if std_r > 1e-6:
                sharpe = round((mean_r / std_r) * math.sqrt(min(252, len(r_multiples))), 2)

        # Ventilation par stratégie
        by_strategy = {}
        strategies = set(str(t.get("strategy", "GENERIC")) for t in trades)
        for st in strategies:
            sub = [t for t in trades if str(t.get("strategy", "GENERIC")) == st]
            sub_wins = [t for t in sub if float(t.get("pnl_usd", 0.0) or 0.0) > 0.0]
            sub_r = [float(t.get("result_r", 0.0) or 0.0) for t in sub]
            by_strategy[st] = {
                "trades": len(sub),
                "win_rate": round(len(sub_wins) / max(len(sub), 1) * 100.0, 1),
                "expectancy_r": round(sum(sub_r) / max(len(sub), 1), 3),
                "net_pnl": round(sum(float(t.get("pnl_usd", 0.0) or 0.0) for t in sub), 2),
            }

        # Ventilation par régime de marché
        by_regime = {}
        regimes = set(str(t.get("market_regime", "UNKNOWN")) for t in trades)
        for reg in regimes:
            sub = [t for t in trades if str(t.get("market_regime", "UNKNOWN")) == reg]
            sub_wins = [t for t in sub if float(t.get("pnl_usd", 0.0) or 0.0) > 0.0]
            sub_r = [float(t.get("result_r", 0.0) or 0.0) for t in sub]
            by_regime[reg] = {
                "trades": len(sub),
                "win_rate": round(len(sub_wins) / max(len(sub), 1) * 100.0, 1),
                "expectancy_r": round(sum(sub_r) / max(len(sub), 1), 3),
            }

        return {
            "total_trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "breakeven": len(be),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy_r": expectancy_r,
            "net_pnl_usd": round(net_pnl, 2),
            "max_drawdown_usd": round(max_dd, 2),
            "sharpe_ratio": sharpe,
            "by_strategy": by_strategy,
            "by_regime": by_regime,
        }
