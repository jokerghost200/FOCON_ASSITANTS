"""
Strategies Package - Inclut les stratégies des 4 traders professionnels (FOCON Strategy Copy).
"""

from src.strategies.base_strategy import BaseTraderStrategy, ProSignal
from src.strategies.corentin_strategy import CorentinStrategy
from src.strategies.benjamin_strategy import BenjaminStrategy
from src.strategies.romain_strategy import RomainStrategy
from src.strategies.nada_strategy import NadaStrategy
from src.strategies.soros_strategy import SorosStrategy
from src.strategies.strategy_profile_manager import StrategyProfileManager

__all__ = [
    "BaseTraderStrategy",
    "ProSignal",
    "CorentinStrategy",
    "BenjaminStrategy",
    "RomainStrategy",
    "NadaStrategy",
    "SorosStrategy",
    "StrategyProfileManager",
]
