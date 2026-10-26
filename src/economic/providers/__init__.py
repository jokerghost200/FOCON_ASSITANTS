# -*- coding: utf-8 -*-
"""
Package FATI Economic Providers (Connecteurs Officiels Données Réelles)
======================================================================
Connecteurs modulaires vers les sources de données macroéconomiques ouvertes et gratuites :
- BLSProvider : Bureau of Labor Statistics (US : Chômage, CPI, Payrolls)
- ECBProvider : Banque Centrale Européenne (BCE : Taux directeurs DFR/MRO/MLFR, HICP)
- EurostatProvider : Eurostat (Zone Euro : IPCH, Chômage harmonisé, PIB)
"""
from __future__ import annotations

from src.economic.providers.base_provider import BaseEconomicProvider
from src.economic.providers.bls_provider import BLSProvider
from src.economic.providers.ecb_provider import ECBProvider
from src.economic.providers.eurostat_provider import EurostatProvider

__all__ = [
    "BaseEconomicProvider",
    "BLSProvider",
    "ECBProvider",
    "EurostatProvider",
]
