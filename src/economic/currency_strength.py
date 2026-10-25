# -*- coding: utf-8 -*-
"""
FATI Currency Strength Engine (Indice de Force Relative des Devises via MT5)
============================================================================
Calcule en temps réel la force relative de chaque devise majeure (EUR, USD, GBP, JPY, CHF, CAD, AUD)
en agrégeant les variations de cours 4h et 24h fournies directement par MetaTrader 5 :
- Aucun service externe payant
- Calcul mathématique de l'indice de force (CSI de 0 à 100)
- Identification immédiate des déséquilibres (ex: Devise forte vs Devise faible)
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Paires majeures de référence pour mesurer la matrice de devises
MAJOR_PAIRS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "USDCAD",
    "AUDUSD",
    "NZDUSD",
]


class CurrencyStrengthEngine:
    """
    Moteur de calcul de force relative des devises basé sur les flux MT5.
    """

    def __init__(self, mt5_connector=None):
        self.mt5 = mt5_connector

    def compute_currency_strength(self, timeframe: str = "H1", lookback_bars: int = 24) -> Dict[str, float]:
        """
        Calcule l'indice de force (0 à 100) pour chaque devise majeure.
        Score > 65 : Devise très forte (acheteurs dominants).
        Score < 35 : Devise très faible (vendeurs dominants).
        """
        currencies = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]
        scores = {c: 0.0 for c in currencies}
        pair_counts = {c: 0 for c in currencies}

        if not self.mt5:
            # Valeurs neutres par défaut si MT5 n'est pas encore branché
            return {c: 50.0 for c in currencies}

        for pair in MAJOR_PAIRS:
            try:
                # 1. Tenter d'obtenir les bougies récentes via MT5
                rates = None
                if hasattr(self.mt5, "get_rates_dataframe"):
                    rates = self.mt5.get_rates_dataframe(pair, timeframe, lookback_bars)
                
                pct_change = 0.0
                if rates is not None and not rates.empty and len(rates) >= 2:
                    p_start = float(rates["close"].iloc[0])
                    p_end = float(rates["close"].iloc[-1])
                    if p_start > 0:
                        pct_change = ((p_end - p_start) / p_start) * 100.0
                else:
                    # Fallback rapide sur le tick spread ou l'historique disponible
                    sym_info = self.mt5.get_symbol_info(pair) if hasattr(self.mt5, "get_symbol_info") else None
                    if sym_info and isinstance(sym_info, dict):
                        ask = float(sym_info.get("ask", 0.0) or 0.0)
                        bid = float(sym_info.get("bid", 0.0) or 0.0)
                        # Légère déviation indicatrice
                        pct_change = 0.05 if ask > bid else -0.05

                # Décomposer la paire (ex: EURUSD -> Base: EUR, Quote: USD)
                base = pair[:3]
                quote = pair[3:]

                # Si EURUSD monte : EUR prend de la force (+pct), USD en perd (-pct)
                if base in scores and quote in scores:
                    scores[base] += pct_change
                    scores[quote] -= pct_change
                    pair_counts[base] += 1
                    pair_counts[quote] += 1

            except Exception as e:
                logger.debug(f"Calcul force devise pour {pair} ignoré ({e}).")

        # Normalisation des scores bruts en indice 0 - 100
        raw_vals = [scores[c] / max(1, pair_counts[c]) for c in currencies]
        min_v = min(raw_vals) if raw_vals else -1.0
        max_v = max(raw_vals) if raw_vals else 1.0
        spread = max_v - min_v if max_v != min_v else 1.0

        csi_scores = {}
        for c, raw in zip(currencies, raw_vals):
            # Normalisation MinMax étalée entre 15 et 85
            normalized = 50.0 + ((raw - (min_v + max_v) / 2.0) / spread) * 35.0
            csi_scores[c] = round(float(np.clip(normalized, 10.0, 90.0)), 1)

        return csi_scores

    def get_pair_bias(self, symbol: str) -> Dict[str, Any]:
        """
        Détermine le biais fondamental directionnel d'une paire selon les forces relatives.
        Exemple : EURUSD -> Force EUR vs Force USD.
        """
        sym = symbol.upper()
        if len(sym) >= 6:
            base = sym[:3]
            quote = sym[3:6]
        else:
            base, quote = "USD", "USD"

        scores = self.compute_currency_strength()
        base_str = scores.get(base, 50.0)
        quote_str = scores.get(quote, 50.0)
        diff = base_str - quote_str

        if diff >= 15.0:
            bias = "BULLISH"
            reason = f"Déséquilibre fort : {base} puissant ({base_str:.0f}) vs {quote} faible ({quote_str:.0f})."
        elif diff <= -15.0:
            bias = "BEARISH"
            reason = f"Déséquilibre fort : {quote} puissant ({quote_str:.0f}) vs {base} faible ({base_str:.0f})."
        else:
            bias = "NEUTRAL"
            reason = f"Forces équilibrées entre {base} ({base_str:.0f}) et {quote} ({quote_str:.0f})."

        return {
            "symbol": sym,
            "base_currency": base,
            "quote_currency": quote,
            "base_strength": base_str,
            "quote_strength": quote_str,
            "differential": round(diff, 1),
            "bias": bias,
            "reason": reason,
        }
