"""
FATI Hypothesis Generator (Générateur d'Hypothèses de Recherche Quantitatives)
=============================================================================
Formule des hypothèses d'amélioration déterministes et testables à partir des données :
- Identification des points faibles (ex: drawdown lors des sessions asiatiques)
- Proposition de modifications ciblées de paramètres (filtre ATR, seuil R:R, session)
- Formalisation sous forme d'expérience isolée avec critères de validation clairs
"""

from typing import List, Dict, Any, Optional
import uuid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ResearchHypothesis:
    """Représente une hypothèse quantitative formalisée prête pour l'expérimentation."""

    def __init__(
        self,
        hypothesis_id: str,
        title: str,
        rationale: str,
        parameter_modifications: Dict[str, Any],
        target_strategy: str = "ALL",
        success_criteria: Optional[Dict[str, Any]] = None,
    ):
        self.hypothesis_id = hypothesis_id or f"hyp_{uuid.uuid4().hex[:8]}"
        self.title = title
        self.rationale = rationale
        self.parameter_modifications = parameter_modifications
        self.target_strategy = target_strategy
        self.success_criteria = success_criteria or {
            "min_expectancy_gain_r": 0.10,
            "max_drawdown_increase_usd": 0.0,
            "min_sample_size": 30,
        }
        self.created_at = datetime.utcnow().isoformat()
        self.status = "PENDING"  # PENDING, TESTING, ACCEPTED, REJECTED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "title": self.title,
            "rationale": self.rationale,
            "parameter_modifications": self.parameter_modifications,
            "target_strategy": self.target_strategy,
            "success_criteria": self.success_criteria,
            "created_at": self.created_at,
            "status": self.status,
        }


class HypothesisGenerator:
    """
    Générateur d'hypothèses autonome basé sur l'analyse des clusters de pertes.
    """

    def generate_from_toxic_patterns(self, toxic_patterns: List[Dict[str, Any]]) -> List[ResearchHypothesis]:
        """
        Déduit des hypothèses testables à partir des patterns toxiques détectés par ExperienceMemory.
        """
        hypotheses = []
        for pat in toxic_patterns:
            st = pat.get("strategy", "GENERIC")
            reg = pat.get("regime", "UNKNOWN")
            sess = pat.get("session", "UNKNOWN")

            title = f"Filtrage conditionnel de {st} en régime {reg} (Session {sess})"
            rationale = (
                f"Le cluster {st}::{reg}::{sess} présente un taux de réussite médiocre de "
                f"{pat.get('win_rate', 0)*100:.1f}% sur {pat.get('sample_size')} trades."
            )
            mods = {
                "disallowed_regimes": [reg],
                "min_confidence_bonus": 10.0,
            }

            hyp = ResearchHypothesis(
                hypothesis_id=f"hyp_{st.lower()}_{reg.lower()[:6]}",
                title=title,
                rationale=rationale,
                parameter_modifications=mods,
                target_strategy=st,
                success_criteria={"min_expectancy_gain_r": 0.15, "min_sample_size": 20},
            )
            hypotheses.append(hyp)

        return hypotheses

    def generate_standard_exploratory_hypotheses(self) -> List[ResearchHypothesis]:
        """Génère un ensemble d'hypothèses quantitatives standard à tester."""
        return [
            ResearchHypothesis(
                hypothesis_id="hyp_rr_tightening",
                title="Exigence de R:R minimal relevé à 2.0 sur setups SMC",
                rationale="Tester si éliminer les R:R < 2.0 améliore l'espérance nette globale malgré moins de trades.",
                parameter_modifications={"min_rr": 2.0},
                target_strategy="SMC",
            ),
            ResearchHypothesis(
                hypothesis_id="hyp_spread_filter_15",
                title="Resserrement du filtre de spread maximal à 15% ATR",
                rationale="Réduire le coût d'exécution toxique sur paires volatiles.",
                parameter_modifications={"max_spread_atr_ratio": 0.15},
                target_strategy="ALL",
            ),
        ]
