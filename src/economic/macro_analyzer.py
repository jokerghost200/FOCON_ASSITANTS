# -*- coding: utf-8 -*-
"""
FATI Macro Analyzer (Analyseur Macroéconomique & Banques Centrales)
===================================================================
Suit les taux directeurs, les postures monétaires (Hawkish/Dovish) et les différentiels
de taux d'intérêt (Carry Trade) pour les 7 grandes devises internationales :
- Réserve Fédérale Américaine (Fed / USD)
- Banque Centrale Européenne (BCE / EUR)
- Banque d'Angleterre (BoE / GBP)
- Banque du Japon (BoJ / JPY)
- Banque Nationale Suisse (BNS / CHF)
- Banque de Réserve d'Australie (RBA / AUD)
- Banque du Canada (BoC / CAD)
"""

import logging
from typing import Dict, Any, Optional

from src.economic.economic_memory import EconomicMemory

logger = logging.getLogger(__name__)

# Base de référence des taux directeurs actuels et postures monétaires des banques centrales
CENTRAL_BANK_RATES = {
    "USD": {
        "bank": "Federal Reserve (Fed)",
        "rate": 5.25,
        "inflation": 2.6,
        "stance": "NEUTRAL_HAWKISH",
        "description": "Taux élevés, politique restrictive dépendante des données d'emploi et d'inflation.",
    },
    "EUR": {
        "bank": "Banque Centrale Européenne (BCE)",
        "rate": 3.25,
        "inflation": 2.2,
        "stance": "DOVISH",
        "description": "Cycle d'assouplissement monétaire pour soutenir la croissance en zone euro.",
    },
    "GBP": {
        "bank": "Bank of England (BoE)",
        "rate": 4.75,
        "inflation": 2.3,
        "stance": "NEUTRAL",
        "description": "Baisse graduelle des taux tout en surveillant la persistance de l'inflation des services.",
    },
    "JPY": {
        "bank": "Bank of Japan (BoJ)",
        "rate": 0.25,
        "inflation": 2.5,
        "stance": "HAWKISH",
        "description": "Sortie des taux négatifs, hausses graduelles pour normaliser la politique monétaire.",
    },
    "CHF": {
        "bank": "Banque Nationale Suisse (BNS)",
        "rate": 1.00,
        "inflation": 1.1,
        "stance": "DOVISH",
        "description": "Inflation très basse, baisses de taux pour freiner l'appréciation excessive du franc suisse.",
    },
    "CAD": {
        "bank": "Bank of Canada (BoC)",
        "rate": 3.75,
        "inflation": 1.9,
        "stance": "DOVISH",
        "description": "Assouplissement monétaire vigoureux face au ralentissement économique.",
    },
    "AUD": {
        "bank": "Reserve Bank of Australia (RBA)",
        "rate": 4.35,
        "inflation": 2.8,
        "stance": "HAWKISH",
        "description": "Prudence maximale face à une inflation sous-jacente tenace, maintien de taux restrictifs.",
    },
}


class MacroAnalyzer:
    """
    Analyseur des déterminants macroéconomiques fondamentaux.
    """

    def __init__(self, memory: Optional[EconomicMemory] = None):
        self.memory = memory or EconomicMemory()
        self._sync_default_rates()

    def _sync_default_rates(self):
        """Enregistre les taux initiaux dans la mémoire SQLite s'ils sont absents."""
        for curr, data in CENTRAL_BANK_RATES.items():
            existing = self.memory.get_latest_macro_snapshot(curr)
            if not existing:
                self.memory.store_macro_snapshot(
                    currency=curr,
                    interest_rate=data["rate"],
                    inflation_rate=data["inflation"],
                    monetary_stance=data["stance"],
                )

    def get_central_bank_info(self, currency: str) -> Dict[str, Any]:
        """Retourne la fiche macroéconomique d'une devise."""
        curr = currency.upper()
        mem_snap = self.memory.get_latest_macro_snapshot(curr)
        ref_data = CENTRAL_BANK_RATES.get(curr, {
            "bank": f"Central Bank of {curr}",
            "rate": 3.0,
            "inflation": 2.0,
            "stance": "NEUTRAL",
            "description": "Données macroéconomiques standard.",
        })

        if mem_snap:
            ref_data = dict(ref_data)
            ref_data["rate"] = float(mem_snap.get("interest_rate", ref_data["rate"]))
            ref_data["inflation"] = float(mem_snap.get("inflation_rate", ref_data["inflation"]))
            ref_data["stance"] = str(mem_snap.get("monetary_stance", ref_data["stance"]))

        return ref_data

    def calculate_interest_differential(self, symbol: str) -> Dict[str, Any]:
        """
        Calcule le différentiel de taux d'intérêt (Carry Trade) pour une paire.
        Ex: EURUSD -> Taux EUR (3.25%) - Taux USD (5.25%) = -2.00% (Défavorable à l'Euro).
        """
        sym = symbol.upper()
        if len(sym) >= 6:
            base = sym[:3]
            quote = sym[3:6]
        else:
            base, quote = "USD", "USD"

        base_info = self.get_central_bank_info(base)
        quote_info = self.get_central_bank_info(quote)

        rate_base = float(base_info.get("rate", 0.0))
        rate_quote = float(quote_info.get("rate", 0.0))
        diff = round(rate_base - rate_quote, 2)

        # Détermination du biais Carry
        if diff >= 1.50:
            carry_bias = "BULLISH"
            reason = f"Différentiel de taux très favorable (+{diff:.2f}%) à {base} ({rate_base}%) vs {quote} ({rate_quote}%)."
        elif diff <= -1.50:
            carry_bias = "BEARISH"
            reason = f"Différentiel de taux défavorable ({diff:.2f}%) à {base} ({rate_base}%) vs {quote} ({rate_quote}%)."
        else:
            carry_bias = "NEUTRAL"
            reason = f"Différentiel de taux modéré ({diff:+.2f}%) entre {base} et {quote}."

        return {
            "symbol": sym,
            "base_currency": base,
            "quote_currency": quote,
            "base_rate": rate_base,
            "quote_rate": rate_quote,
            "differential": diff,
            "carry_bias": carry_bias,
            "base_stance": base_info.get("stance"),
            "quote_stance": quote_info.get("stance"),
            "reason": reason,
        }
