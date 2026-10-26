# -*- coding: utf-8 -*-
"""
Package FATI Economic & Macro Intelligence (Système Économique FOCON)
====================================================================
Regroupe les composants d'intelligence fondamentale et macroéconomique :
- EconomicMemory : Mémoire relationnelle locale SQLite
- CalendarProvider : Flux du calendrier économique gratuit et récurrence hors-ligne
- NewsCollector : Collecteur RSS et analyseur de sentiment local
- CurrencyStrengthEngine : Force relative des devises (CSI) via MT5
- MacroAnalyzer : Taux directeurs et différentiels de taux des banques centrales
- EventImpactEngine : Surprise économique et impact sur les paires
"""

from src.economic.economic_memory import EconomicMemory
from src.economic.calendar_provider import CalendarProvider
from src.economic.news_collector import NewsCollector
from src.economic.currency_strength import CurrencyStrengthEngine
from src.economic.macro_analyzer import MacroAnalyzer
from src.economic.event_impact import EventImpactEngine
from src.economic.providers.bls_provider import BLSProvider
from src.economic.providers.ecb_provider import ECBProvider
from src.economic.providers.eurostat_provider import EurostatProvider

__all__ = [
    "EconomicMemory",
    "CalendarProvider",
    "NewsCollector",
    "CurrencyStrengthEngine",
    "MacroAnalyzer",
    "EventImpactEngine",
    "BLSProvider",
    "ECBProvider",
    "EurostatProvider",
]
