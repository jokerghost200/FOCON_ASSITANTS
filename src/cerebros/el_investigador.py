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

        # 1. Récupération des trades fermés récents (triés strictement par ordre chronologique ASC)
        trades = []
        if sabio and hasattr(sabio, "journal"):
            raw_trades = sabio.journal.get_closed_trades(limit=100) or []
            trades = sorted(raw_trades, key=lambda t: (str(t.get("timestamp") or ""), int(t.get("id") or 0)))

        # 2. Séparation stricte In-Sample (50% anciens pour découverte) / Out-of-Sample (50% récents pour validation)
        toxic = []
        if len(trades) >= 20:
            split_idx = max(10, len(trades) // 2)
            in_sample_trades = trades[:split_idx]
            # Détection de patterns toxiques uniquement sur la fenêtre In-Sample (zéro fuite du futur)
            regime_stats: Dict[str, List[float]] = {}
            for tr in in_sample_trades:
                reg = str(tr.get("regime") or "UNKNOWN")
                regime_stats.setdefault(reg, []).append(float(tr.get("pnl_r", 0.0) or 0.0))
            for reg, pnls in regime_stats.items():
                if len(pnls) >= 3:
                    wins = sum(1 for p in pnls if p > 0)
                    wr = wins / len(pnls)
                    if wr <= 0.35:
                        toxic.append({"regime": reg, "sample_size": len(pnls), "win_rate": round(wr, 3)})
        elif sabio and hasattr(sabio, "detectar_patrones_toxicos"):
            toxic = sabio.detectar_patrones_toxicos(min_sample=3, max_win_rate=0.35)

        hypotheses = self.generar_hipotesis(toxic if toxic else None)
        report["hypotheses_generated"] = len(hypotheses)
        logger.info(f"💡 [El Investigador] {len(hypotheses)} hypothèse(s) formulée(s) (ordre chronologique strict).")

        # 3. Test des hypothèses sur le jeu de données si suffisant (séparation stricte In-Sample / Out-of-Sample)
        if len(trades) >= 20:
            split_idx = max(10, len(trades) // 2)
            in_sample_trades = trades[:split_idx]
            out_of_sample_trades = trades[split_idx:]
            for hyp in hypotheses[:2]:  # Test ciblé des meilleures hypothèses
                variant_in_sample = []
                variant_all_chronological = []
                disallowed = hyp.parameter_modifications.get("disallowed_regimes", [])
                min_conf = float(hyp.parameter_modifications.get("min_confidence", 0.0) or 0.0)
                min_rr = float(hyp.parameter_modifications.get("min_rr", 0.0) or 0.0)
                friction_r = float(hyp.parameter_modifications.get("friction_penalty_r", 0.0) or 0.0)

                for idx_t, tr in enumerate(trades):
                    # Simulation honnête et symétrique (AUCUN cherry-picking rétrospectif) :
                    # Si une règle exclut un contexte, elle exclut TOUS les trades de ce contexte (gains ET pertes)
                    if disallowed and tr.get("regime") in disallowed:
                        continue
                    if min_conf > 0 and float(tr.get("confidence", 0.0) or 0.0) < min_conf:
                        continue
                    if min_rr > 0 and float(tr.get("rr", 0.0) or 0.0) < min_rr:
                        continue

                    v_tr = dict(tr)
                    if friction_r > 0:
                        v_tr["pnl_r"] = round(float(v_tr.get("pnl_r", 0.0) or 0.0) - friction_r, 4)

                    variant_all_chronological.append(v_tr)
                    if idx_t < split_idx:
                        variant_in_sample.append(v_tr)

                # Sélection/évaluation d'expérience sur In-Sample, validation aveugle Walk-Forward chronologique
                exp_result = self.ejecutar_experimento(hyp, in_sample_trades, variant_in_sample)
                wf_result = self.validar_walk_forward(variant_all_chronological)
                report["experiments_run"] += 1
                report["in_sample_size"] = len(in_sample_trades)
                report["out_of_sample_size"] = len(out_of_sample_trades)

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
