# -*- coding: utf-8 -*-
"""
LOS CINCO CEREBROS DE FOCON ASSISTANT (FATI)
============================================
Architecture modulaire des 5 cerveaux spécialisés :

1. ElObservador     (src.cerebros.el_observador)     : Les Cerveaux Observateurs (OBSERVER)
2. ElEstratega      (src.cerebros.el_estratega)      : Les Cerveaux de Décision Stratégique (DÉCIDER)
3. ElSabio          (src.cerebros.el_sabio)          : Les Cerveaux d'Apprentissage & Mémoire (APPRENDRE)
4. ElInvestigador   (src.cerebros.el_investigador)   : Le Cerveau de Recherche & Sandbox (EXPÉRIMENTER)
5. ElCentinela      (src.cerebros.el_centinela)      : Le Cerveau Sentinelle & Protection (SÉCURISER)
"""

from src.cerebros.el_observador import ElObservador
from src.cerebros.el_estratega import ElEstratega
from src.cerebros.el_sabio import ElSabio
from src.cerebros.el_investigador import ElInvestigador
from src.cerebros.el_centinela import ElCentinela

__all__ = [
    "ElObservador",
    "ElEstratega",
    "ElSabio",
    "ElInvestigador",
    "ElCentinela",
]
