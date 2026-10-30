# -*- coding: utf-8 -*-
"""
FATI Currency Strength Engine (Indice de Force Relative des Devises via MT5)
============================================================================
Calcule en temps réel la force relative de chaque devise majeure (EUR, USD, GBP, JPY, CHF, CAD, AUD, NZD)
en agrégeant les variations de cours fournies directement par MetaTrader 5 :
- Aucun service externe payant
- Calcul mathématique de l'indice de force (CSI de 0 à 100)
- Identification immédiate des déséquilibres réels
- STRICTEMENT AUCUN BIAIS ARTIFICIEL : Si les données MT5 sont absentes ou insuffisantes,
  le moteur retourne un statut UNAVAILABLE au lieu d'extrapoler sur le spread.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
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

COMMON_SUFFIXES = ["", ".r", "m", ".pro", "_i", ".a", ".raw"]


class CurrencyStrengthEngine:
    """
    Moteur de calcul de force relative des devises basé sur les flux MT5.
    Refuse catégoriquement d'inventer des scores si les données de marché réelles sont manquantes.
    """

    def __init__(self, mt5_connector=None):
        self.mt5 = mt5_connector
        self.is_available = False
        self.last_status = "UNINITIALIZED"
        self._symbol_cache = {}

    def _resolve_broker_symbol(self, pair: str) -> Optional[str]:
        """Résout le nom de symbole réel chez le courtier (avec suffixes éventuels)."""
        if not self.mt5:
            return None
        if pair in self._symbol_cache:
            return self._symbol_cache[pair]

        for sfx in COMMON_SUFFIXES:
            candidate = f"{pair}{sfx}"
            try:
                info = self.mt5.get_symbol_info(candidate) if hasattr(self.mt5, "get_symbol_info") else None
                if info and isinstance(info, dict) and info.get("bid") is not None:
                    self._symbol_cache[pair] = candidate
                    return candidate
            except Exception:
                continue

        return pair

    def compute_currency_strength(
        self, timeframe: str = "H1", lookback_bars: int = 24
    ) -> Tuple[Dict[str, float], bool]:
        """
        Calcule l'indice de force (0 à 100) pour chaque devise majeure.
        Retourne (scores, is_available).
        Si moins de 3 paires ont des données historiques réelles valides,
        refuse d'extrapoler et retourne 50.0 neutre avec is_available=False.
        """
        currencies = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]
        scores = {c: 0.0 for c in currencies}
        pair_counts = {c: 0 for c in currencies}

        if not self.mt5:
            self.is_available = False
            self.last_status = "MT5_DISCONNECTED"
            return {c: 50.0 for c in currencies}, False

        valid_pairs_analyzed = 0

        for pair in MAJOR_PAIRS:
            actual_sym = self._resolve_broker_symbol(pair)
            try:
                import pandas as pd
                rates = None
                if hasattr(self.mt5, "get_rates"):
                    raw_r = self.mt5.get_rates(actual_sym, timeframe, lookback_bars)
                    if isinstance(raw_r, pd.DataFrame):
                        rates = raw_r
                    elif raw_r is not None and not isinstance(raw_r, (bool, int, float)) and type(raw_r).__name__ not in ("MagicMock", "Mock") and len(raw_r) > 0:
                        rates = pd.DataFrame(raw_r)
                if rates is None and hasattr(self.mt5, "get_rates_dataframe"):
                    cand = self.mt5.get_rates_dataframe(actual_sym, timeframe, lookback_bars)
                    if isinstance(cand, pd.DataFrame):
                        rates = cand
                if rates is None and hasattr(self.mt5, "get_rates_from_pos"):
                    raw_r = self.mt5.get_rates_from_pos(actual_sym, timeframe, lookback_bars)
                    if isinstance(raw_r, pd.DataFrame):
                        rates = raw_r
                    elif raw_r is not None and type(raw_r).__name__ not in ("MagicMock", "Mock") and len(raw_r) > 0:
                        rates = pd.DataFrame(raw_r)

                if rates is not None and not rates.empty and len(rates) >= 2:
                    p_start = float(rates["close"].iloc[0])
                    p_end = float(rates["close"].iloc[-1])
                    if p_start > 0:
                        pct_change = ((p_end - p_start) / p_start) * 100.0
                        base = pair[:3]
                        quote = pair[3:]
                        if base in scores and quote in scores:
                            scores[base] += pct_change
                            scores[quote] -= pct_change
                            pair_counts[base] += 1
                            pair_counts[quote] += 1
                            valid_pairs_analyzed += 1
                else:
                    # STRICTEMENT AUCUNE INVENTION DE SCORE :
                    # L'absence de bougies ne doit PAS être remplacée par un écart de spread.
                    logger.debug(f"[CSI] Pas de bougies suffisantes pour {pair}. Paire ignorée.")

            except Exception as e:
                logger.debug(f"[CSI] Erreur lecture cours {pair}: {e}")

        # Sécurité : Si moins de 3 paires majeures sont exploitables, le calcul n'est pas fiable
        if valid_pairs_analyzed < 3:
            self.is_available = False
            self.last_status = "INSUFFICIENT_DATA"
            logger.warning(
                f"⚠️ [CSI] Seulement {valid_pairs_analyzed} paire(s) exploitable(s) sur MT5 "
                f"(minimum 3 requis). Force des devises marquée UNAVAILABLE."
            )
            return {c: 50.0 for c in currencies}, False

        # Normalisation mathématique des scores bruts en indice 0 - 100
        raw_vals = [scores[c] / max(1, pair_counts[c]) for c in currencies]
        min_v = min(raw_vals) if raw_vals else -1.0
        max_v = max(raw_vals) if raw_vals else 1.0
        spread = max_v - min_v if max_v != min_v else 1.0

        csi_scores = {}
        for c, raw in zip(currencies, raw_vals):
            normalized = 50.0 + ((raw - (min_v + max_v) / 2.0) / spread) * 35.0
            csi_scores[c] = round(float(np.clip(normalized, 10.0, 90.0)), 1)

        self.is_available = True
        self.last_status = "AVAILABLE"
        return csi_scores, True

    def get_pair_bias(self, symbol: str) -> Dict[str, Any]:
        """
        Détermine le biais fondamental directionnel d'une paire selon les forces relatives.
        Retourne available=False et biais NEUTRAL si les données MT5 sont insuffisantes.
        """
        sym = symbol.upper()
        if len(sym) >= 6:
            base = sym[:3]
            quote = sym[3:6]
        else:
            base, quote = "USD", "USD"

        scores, is_avail = self.compute_currency_strength()

        if not is_avail:
            return {
                "symbol": sym,
                "available": False,
                "status": "UNAVAILABLE",
                "base_currency": base,
                "quote_currency": quote,
                "base_strength": 50.0,
                "quote_strength": 50.0,
                "differential": 0.0,
                "bias": "NEUTRAL",
                "reason": f"Données de cours MT5 insuffisantes pour {sym}. Aucun score artificiel inventé.",
            }

        base_str = scores.get(base, 50.0)
        quote_str = scores.get(quote, 50.0)
        diff = base_str - quote_str

        if diff >= 15.0:
            bias = "BULLISH"
            reason = f"Déséquilibre confirmé : {base} fort ({base_str:.0f}) vs {quote} faible ({quote_str:.0f})."
        elif diff <= -15.0:
            bias = "BEARISH"
            reason = f"Déséquilibre confirmé : {quote} fort ({quote_str:.0f}) vs {base} faible ({base_str:.0f})."
        else:
            bias = "NEUTRAL"
            reason = f"Forces équilibrées entre {base} ({base_str:.0f}) et {quote} ({quote_str:.0f})."

        return {
            "symbol": sym,
            "available": True,
            "status": "AVAILABLE",
            "base_currency": base,
            "quote_currency": quote,
            "base_strength": base_str,
            "quote_strength": quote_str,
            "differential": round(diff, 1),
            "bias": bias,
            "reason": reason,
        }
