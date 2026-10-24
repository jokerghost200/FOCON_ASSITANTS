"""
FATI Walk-Forward Validator (Validation Chronologique Hors-Échantillon)
======================================================================
Élimine le sur-ajustement (Overfitting / Curve-Fitting) en testant rigoureusement :
- Période d'optimisation / In-Sample (70%)
- Période de validation aveugle / Out-of-Sample (30%)
- Calcul du Walk-Forward Efficiency Ratio (WFE Ratio)
"""

from typing import List, Dict, Any
import logging
from src.learning.performance_analyzer import PerformanceAnalyzer

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Validateur scientifique hors-échantillon.
    Garantit qu'une stratégie fonctionne sur des données qu'elle n'a jamais vues.
    """

    def __init__(
        self,
        split_ratio: float = 0.70,
        train_ratio: Optional[float] = None,
        min_oos_trades: int = 10,
        min_wfe_ratio: float = 0.50,
    ):
        self.split_ratio = train_ratio if train_ratio is not None else split_ratio
        self.min_oos_trades = min_oos_trades
        self.min_wfe_ratio = min_wfe_ratio

    def validate_walk_forward(self, chronological_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Alias pour validate_dataset."""
        return self.validate_dataset(chronological_trades)

    def validate_dataset(self, chronological_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Découpe chronologiquement les trades et vérifie la persistance de l'edge.
        """
        total = len(chronological_trades)
        if total < (self.min_oos_trades * 2):
            return {
                "is_validated": False,
                "reason": f"Nombre total de trades insuffisant pour Walk-Forward ({total} < {self.min_oos_trades*2}).",
                "wfe_ratio": 0.0,
                "in_sample_trades": 0,
                "out_sample_trades": 0,
            }

        split_idx = int(total * self.split_ratio)
        in_sample = chronological_trades[:split_idx]
        out_of_sample = chronological_trades[split_idx:]

        analyzer = PerformanceAnalyzer()
        perf_is = analyzer.analyze_trades(in_sample)
        perf_oos = analyzer.analyze_trades(out_of_sample)

        exp_is = perf_is.get("expectancy_r", 0.0)
        exp_oos = perf_oos.get("expectancy_r", 0.0)

        # Calcul du Walk-Forward Efficiency Ratio
        if exp_is > 0:
            wfe = round(exp_oos / exp_is, 2)
        else:
            wfe = 0.0

        # Critère d'acceptation : L'OOS doit impérativement avoir une espérance positive et préserver au moins 50% de l'IS
        is_robust = (exp_oos > 0.05) and (wfe >= self.min_wfe_ratio) and (perf_oos.get("win_rate", 0) >= 40.0)

        verdict = (
            f"CERTIFIE ROBUSTE : Edge persistant hors-échantillon (OOS Exp={exp_oos:+.2f}R, WFE={wfe*100:.0f}%)"
            if is_robust
            else f"REJETE (Sur-ajustement) : Dégradation hors-échantillon (OOS Exp={exp_oos:+.2f}R vs IS Exp={exp_is:+.2f}R, WFE={wfe*100:.0f}%)"
        )

        return {
            "is_validated": is_robust,
            "verdict": verdict,
            "wfe_ratio": wfe,
            "in_sample_trades": len(in_sample),
            "out_sample_trades": len(out_of_sample),
            "in_sample_metrics": perf_is,
            "out_of_sample_metrics": perf_oos,
        }
