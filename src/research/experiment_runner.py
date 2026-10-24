"""
FATI Experiment Runner (Moteur d'Expérimentation Quantitatif Isolé)
==================================================================
Exécute des tests A/B scientifiques dans un bac à sable (Sandbox) isolé :
- Rejoue des données historiques ou simulations avec les mêmes hypothèses de coûts
- Compare rigoureusement la version de base (Baseline) et la variante (Variant)
- Produit un rapport diagnostique statistique d'arbitrage
"""

from typing import Dict, Any, List, Optional
import logging
from src.research.hypothesis_generator import ResearchHypothesis
from src.learning.performance_analyzer import PerformanceAnalyzer

logger = logging.getLogger(__name__)


class ExperimentResult:
    def __init__(
        self,
        hypothesis_id: str,
        baseline_metrics: Dict[str, Any],
        variant_metrics: Dict[str, Any],
        is_statistically_superior: bool,
        expectancy_delta_r: float,
        drawdown_delta_usd: float,
        verdict: str,
    ):
        self.hypothesis_id = hypothesis_id
        self.baseline_metrics = baseline_metrics
        self.variant_metrics = variant_metrics
        self.is_statistically_superior = is_statistically_superior
        self.expectancy_delta_r = expectancy_delta_r
        self.drawdown_delta_usd = drawdown_delta_usd
        self.verdict = verdict

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "is_statistically_superior": self.is_statistically_superior,
            "expectancy_delta_r": self.expectancy_delta_r,
            "drawdown_delta_usd": self.drawdown_delta_usd,
            "verdict": self.verdict,
            "baseline": self.baseline_metrics,
            "variant": self.variant_metrics,
        }


class ExperimentRunner:
    """
    Exécute des expériences comparatives isolées.
    Ne modifie JAMAIS la configuration de production directement.
    """

    def run_comparison(
        self,
        hypothesis: ResearchHypothesis,
        baseline_trades: List[Dict[str, Any]],
        simulated_variant_trades: List[Dict[str, Any]],
    ) -> ExperimentResult:
        """
        Compare les métriques de la variante face à la baseline sur un même échantillon.
        """
        analyzer = PerformanceAnalyzer()
        base_perf = analyzer.analyze_trades(baseline_trades)
        var_perf = analyzer.analyze_trades(simulated_variant_trades)

        base_exp = base_perf.get("expectancy_r", 0.0)
        var_exp = var_perf.get("expectancy_r", 0.0)
        exp_delta = round(var_exp - base_exp, 3)

        base_dd = base_perf.get("max_drawdown_usd", 0.0)
        var_dd = var_perf.get("max_drawdown_usd", 0.0)
        dd_delta = round(var_dd - base_dd, 2)

        crit = hypothesis.success_criteria
        min_gain = float(crit.get("min_expectancy_gain_r", 0.05))
        max_dd_inc = float(crit.get("max_drawdown_increase_usd", 0.0))

        is_superior = (exp_delta >= min_gain) and (dd_delta <= max_dd_inc) and (var_perf.get("total_trades", 0) >= 15)

        if is_superior:
            verdict = f"VALIDE: L'hypothèse apporte un gain de +{exp_delta:.2f}R sans dégradation du drawdown."
            hypothesis.status = "ACCEPTED"
        else:
            verdict = f"REJETE: Pas d'avantage statistique prouvé (Delta={exp_delta:+.2f}R, seuil requis={min_gain:.2f}R)."
            hypothesis.status = "REJECTED"

        return ExperimentResult(
            hypothesis_id=hypothesis.hypothesis_id,
            baseline_metrics=base_perf,
            variant_metrics=var_perf,
            is_statistically_superior=is_superior,
            expectancy_delta_r=exp_delta,
            drawdown_delta_usd=dd_delta,
            verdict=verdict,
        )
