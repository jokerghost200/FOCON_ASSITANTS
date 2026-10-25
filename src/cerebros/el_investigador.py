# -*- coding: utf-8 -*-
"""
CEREBRO 4 : EL INVESTIGADOR (Pilier 4 - EXPÉRIMENTER & RECHERCHER)
==================================================================
Nom Espagnol : El Investigador ("Le Chercheur / L'Investigateur")
Rôle : Explore, formule des hypothèses scientifiques, teste de nouvelles configurations
en sandbox isolé et valide la robustesse hors-échantillon (Walk-Forward) sans risque pour le capital.

Composants fédérés :
- HypothesisGenerator : Formulation d'hypothèses quantitatives ciblées
- ExperimentRunner : Exécution d'expériences A/B comparatives dans un bac à sable isolé
- WalkForwardValidator : Validation chronologique hors-échantillon (In-Sample / Out-of-Sample)
- ModelRegistry : Registre versionné de tous les modèles, configurations et hyperparamètres
"""

import logging
from typing import Dict, Any, List, Optional

from src.research.hypothesis_generator import HypothesisGenerator, ResearchHypothesis
from src.research.experiment_runner import ExperimentRunner, ExperimentResult
from src.research.walk_forward import WalkForwardValidator
from src.research.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ElInvestigador:
    """
    Cerveau Chercheur (El Investigador)
    Conduit la recherche quantitative autonome et teste les améliorations
    dans un environnement étanche avant toute considération pour la production.
    """

    def __init__(
        self,
        hypotheses: Optional[HypothesisGenerator] = None,
        experiments: Optional[ExperimentRunner] = None,
        walk_forward: Optional[WalkForwardValidator] = None,
        registry: Optional[ModelRegistry] = None,
    ):
        self.hypotheses = hypotheses or HypothesisGenerator()
        self.experiments = experiments or ExperimentRunner()
        self.walk_forward = walk_forward or WalkForwardValidator()
        self.registry = registry or ModelRegistry()

    def generar_hipotesis(
        self, toxic_patterns: Optional[List[Dict[str, Any]]] = None
    ) -> List[ResearchHypothesis]:
        """
        Formule des hypothèses quantitatives à partir de clusters toxiques ou de règles exploratoires.
        """
        if toxic_patterns:
            return self.hypotheses.generate_from_toxic_patterns(toxic_patterns)
        return self.hypotheses.generate_standard_exploratory_hypotheses()

    def ejecutar_experimento(
        self,
        hypothesis: ResearchHypothesis,
        baseline_trades: List[Dict[str, Any]],
        simulated_variant_trades: List[Dict[str, Any]],
    ) -> ExperimentResult:
        """
        Exécute une expérience A/B isolée entre la version de base et la variante candidate.
        """
        return self.experiments.run_comparison(
            hypothesis=hypothesis,
            baseline_trades=baseline_trades,
            simulated_variant_trades=simulated_variant_trades,
        )

    def validar_walk_forward(
        self, chronological_trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Vérifie la robustesse hors-échantillon (Walk-Forward) pour éliminer le sur-ajustement.
        """
        return self.walk_forward.validate_walk_forward(chronological_trades)

    def registrar_modelo(
        self,
        version: str,
        name: str,
        parameters: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        status: str = "CANDIDATE",
    ) -> bool:
        """
        Enregistre une nouvelle version de modèle ou de configuration dans le registre central.
        """
        return self.registry.register_model(
            version=version,
            name=name,
            parameters=parameters,
            metrics=metrics or {},
            status=status,
        )

    def obtener_modelo_activo(self) -> Optional[Dict[str, Any]]:
        """Retourne la configuration du modèle actuellement actif."""
        return self.registry.get_active_model()
