"""
Package de recherche et validation scientifique FATI.
Fournit :
- HypothesisGenerator / ResearchHypothesis : Formulateur d'hypothèses
- ExperimentRunner / ExperimentResult : Bac à sable expérimental A/B
- WalkForwardValidator : Validation chronologique hors-échantillon
- ModelRegistry : Gestion du cycle de vie et versionnage des modèles
"""
from src.research.hypothesis_generator import HypothesisGenerator, ResearchHypothesis
from src.research.experiment_runner import ExperimentRunner, ExperimentResult
from src.research.walk_forward import WalkForwardValidator
from src.research.model_registry import ModelRegistry

__all__ = [
    "HypothesisGenerator",
    "ResearchHypothesis",
    "ExperimentRunner",
    "ExperimentResult",
    "WalkForwardValidator",
    "ModelRegistry",
]
