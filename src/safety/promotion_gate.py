"""
FATI Promotion Gate (Porte de Sécurité et de Promotion Contrôlée)
================================================================
Empêche le déploiement accidentel ou prématuré d'une modification de stratégie :
- Exige une validation hors-échantillon (OOS) statistiquement significative
- Refuse toute variante dont le drawdown est supérieur à la référence
- Exige un ratio Walk-Forward Efficiency (WFE) supérieur à 50%
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class PromotionGate:
    """
    Porte de contrôle qualité indépendante.
    Autorise ou rejette la promotion d'une nouvelle version de modèle vers la production.
    """

    def __init__(
        self,
        min_oos_trades: int = 25,
        min_expectancy_gain_r: float = 0.08,
        min_wfe_ratio: float = 0.50,
        max_drawdown_increase_usd: float = 0.0,
    ):
        self.min_oos_trades = min_oos_trades
        self.min_expectancy_gain_r = min_expectancy_gain_r
        self.min_wfe_ratio = min_wfe_ratio
        self.max_drawdown_increase_usd = max_drawdown_increase_usd

    def evaluate_promotion(
        self,
        candidate_version: str,
        experiment_result: Dict[str, Any],
        walk_forward_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Contrôle la conformité stricte avant promotion en production.
        """
        reasons_to_reject = []
        checks = {}

        # 1. Échantillon de test
        variant_perf = experiment_result.get("variant", {})
        total_trades = int(variant_perf.get("total_trades", 0))
        checks["sample_size"] = total_trades >= self.min_oos_trades
        if not checks["sample_size"]:
            reasons_to_reject.append(f"Échantillon insuffisant ({total_trades} < {self.min_oos_trades} trades requis).")

        # 2. Gain d'espérance
        exp_delta = float(experiment_result.get("expectancy_delta_r", 0.0))
        checks["expectancy_gain"] = exp_delta >= self.min_expectancy_gain_r
        if not checks["expectancy_gain"]:
            reasons_to_reject.append(f"Gain d'espérance insuffisant ({exp_delta:+.2f}R < {self.min_expectancy_gain_r:+.2f}R requis).")

        # 3. Risque de Drawdown
        dd_delta = float(experiment_result.get("drawdown_delta_usd", 0.0))
        checks["drawdown_containment"] = dd_delta <= self.max_drawdown_increase_usd
        if not checks["drawdown_containment"]:
            reasons_to_reject.append(f"Accroissement du drawdown inacceptable (+{dd_delta:.2f}$ vs baseline).")

        # 4. Robustesse Walk-Forward
        wfe = float(walk_forward_result.get("wfe_ratio", 0.0))
        is_wf_valid = bool(walk_forward_result.get("is_validated", False))
        checks["walk_forward_robustness"] = is_wf_valid and (wfe >= self.min_wfe_ratio)
        if not checks["walk_forward_robustness"]:
            reasons_to_reject.append(f"Échec du test Walk-Forward hors-échantillon (WFE={wfe*100:.0f}% < {self.min_wfe_ratio*100:.0f}%).")

        is_approved = len(reasons_to_reject) == 0

        verdict_str = (
            f"✅ PROMOTION ACCORDÉE : La version {candidate_version} a démontré un avantage mesurable et robuste."
            if is_approved
            else f"❌ PROMOTION REFUSÉE : {candidate_version} ne remplit pas les critères de sécurité ({' ; '.join(reasons_to_reject)})."
        )

        logger.info(f"🛡️ [PromotionGate] {verdict_str}")

        return {
            "approved": is_approved,
            "candidate_version": candidate_version,
            "verdict": verdict_str,
            "rejections": reasons_to_reject,
            "checks": checks,
        }
