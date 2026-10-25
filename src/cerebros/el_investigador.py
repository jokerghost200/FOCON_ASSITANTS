# -*- coding: utf-8 -*-
"""
CEREBRO 4 : EL INVESTIGADOR (Pilier 4 - EXPÉRIMENTER & RECHERCHER)
==================================================================
Nom Espagnol : El Investigador ("Le Chercheur / L'Investigateur")
Rôle : Explore, formule des hypothèses scientifiques, teste de nouvelles configurations
en sandbox isolé et valide la robustesse hors-échantillon (Walk-Forward) sans risque pour le capital.
Ne modifie JAMAIS la production sans validation indépendante par El Centinela.

Composants fédérés :
- HypothesisGenerator : Formulation d'hypothèses quantitatives ciblées
- ExperimentRunner : Exécution d'expériences A/B comparatives dans un bac à sable isolé
- WalkForwardValidator : Validation chronologique hors-échantillon (In-Sample / Out-of-Sample)
- ModelRegistry : Registre versionné de tous les modèles, configurations et hyperparamètres
"""

import logging
from typing import Dict, Any, List, Optional
import uuid

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

    def ejecutar_ciclo_investigacion(self, sabio=None, centinela=None) -> Dict[str, Any]:
        """
        Cycle de recherche et validation scientifique autonome :
        1. Analyse des pertes passées issues d'El Sabio
        2. Détection des clusters toxiques et formulation d'hypothèses
        3. Exécution d'expériences en Sandbox
        4. Validation aveugle Walk-Forward
        5. Soumission stricte au sas de sécurité d'El Centinela (ne touche JAMAIS à la prod sans son accord)
        """
        logger.info("🔬 [El Investigador] Lancement du cycle de recherche quantitative autonome...")
        report = {
            "hypotheses_generated": 0,
            "experiments_run": 0,
            "promotions_evaluated": 0,
            "status": "COMPLETED",
        }

        # 1. Récupération des trades fermés récents
        trades = []
        if sabio and hasattr(sabio, "journal"):
            trades = sabio.journal.get_closed_trades(limit=100) or []

        # 2. Détection de patterns toxiques
        toxic = []
        if sabio and hasattr(sabio, "detectar_patrones_toxicos"):
            toxic = sabio.detectar_patrones_toxicos(min_sample=3, max_win_rate=0.35)

        hypotheses = self.generar_hipotesis(toxic if toxic else None)
        report["hypotheses_generated"] = len(hypotheses)
        logger.info(f"💡 [El Investigador] {len(hypotheses)} hypothèse(s) formulée(s).")

        # 3. Test des hypothèses sur le jeu de données si suffisant
        if len(trades) >= 20:
            for hyp in hypotheses[:2]:  # Test ciblé des meilleures hypothèses
                variant_trades = []
                for tr in trades:
                    # Simulation d'application des modifications
                    v_tr = dict(tr)
                    if tr.get("pnl_usd", 0) < 0 and tr.get("regime") in hyp.parameter_modifications.get("disallowed_regimes", []):
                        continue  # Le filtre aurait évité cette perte
                    variant_trades.append(v_tr)

                exp_result = self.ejecutar_experimento(hyp, trades, variant_trades)
                wf_result = self.validar_walk_forward(variant_trades)
                report["experiments_run"] += 1

                cand_ver = f"v_{hyp.hypothesis_id}_{uuid.uuid4().hex[:4]}"
                if centinela and hasattr(centinela, "verificar_promocion"):
                    promo_eval = centinela.verificar_promocion(cand_ver, exp_result.to_dict(), wf_result)
                    report["promotions_evaluated"] += 1
                    status = "CANDIDATE_APPROVED" if promo_eval.get("approved") else "REJECTED"
                    self.registrar_modelo(
                        version=cand_ver,
                        name=hyp.title,
                        parameters=hyp.parameter_modifications,
                        metrics={"expectancy_delta_r": exp_result.expectancy_delta_r, "promo_eval": promo_eval},
                        status=status,
                    )
                    logger.info(f"⚖️ [El Investigador + El Centinela] Résultat promotion {cand_ver}: {status}")

        return report
