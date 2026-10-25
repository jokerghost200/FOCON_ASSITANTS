# -*- coding: utf-8 -*-
"""
FATI Event Impact Engine (Moteur d'Impact des Publications Économiques)
======================================================================
Évalue l'écart entre les chiffres publiés ("Actual") et le consensus ("Forecast") :
- Calcul du coefficient d'écart économique (Economic Surprise Index)
- Déduction de la pression directionnelle sur la devise concernée
- Mise en corrélation avec les réactions historiques mémorisées
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class EventImpactEngine:
    """
    Moteur quantitatif d'évaluation de l'impact des chiffres économiques.
    """

    def evaluate_surprise(
        self,
        event_title: str,
        currency: str,
        actual: Optional[float],
        forecast: Optional[float],
        previous: Optional[float],
    ) -> Dict[str, Any]:
        """
        Calcule la surprise économique et le biais résultant pour la devise.
        """
        if actual is None or forecast is None:
            return {
                "event_title": event_title,
                "currency": currency,
                "surprise_score": 0.0,
                "verdict": "NO_SURPRISE_DATA",
                "currency_bias": "NEUTRAL",
                "reason": "Chiffres réels ou prévisions non encore disponibles.",
            }

        diff = actual - forecast
        ref_base = abs(forecast) if abs(forecast) > 1e-4 else 1.0
        surprise_pct = (diff / ref_base) * 100.0

        title_lower = event_title.lower()

        # Règle inversée pour le chômage (Unemployment) : Chômage en baisse = Positif pour la devise !
        is_inverted = any(k in title_lower for k in ("unemployment", "jobless", "claims"))

        if is_inverted:
            effective_diff = -diff
        else:
            effective_diff = diff

        if effective_diff > 0:
            if abs(surprise_pct) >= 15.0 or abs(diff) >= 0.2:
                verdict = "STRONG_POSITIVE_BEAT"
                bias = "BULLISH"
            else:
                verdict = "SLIGHT_POSITIVE_BEAT"
                bias = "MILDLY_BULLISH"
            reason = f"Publication supérieure aux attentes ({actual} vs {forecast} prévu). Favorable à {currency}."
        elif effective_diff < 0:
            if abs(surprise_pct) >= 15.0 or abs(diff) >= 0.2:
                verdict = "STRONG_NEGATIVE_MISS"
                bias = "BEARISH"
            else:
                verdict = "SLIGHT_NEGATIVE_MISS"
                bias = "MILDLY_BEARISH"
            reason = f"Publication inférieure aux attentes ({actual} vs {forecast} prévu). Défavorable à {currency}."
        else:
            verdict = "IN_LINE_WITH_CONSENSUS"
            bias = "NEUTRAL"
            reason = f"Publication conforme aux prévisions ({actual} = {forecast})."

        return {
            "event_title": event_title,
            "currency": currency.upper(),
            "actual": actual,
            "forecast": forecast,
            "previous": previous,
            "diff": round(diff, 3),
            "surprise_pct": round(surprise_pct, 1),
            "verdict": verdict,
            "currency_bias": bias,
            "reason": reason,
        }

    def map_surprise_to_symbol_impact(self, symbol: str, surprise_result: Dict[str, Any]) -> str:
        """
        Traduit le biais sur une devise en impact sur une paire de trading.
        Ex: Biais BULLISH sur USD -> BEARISH sur EURUSD et XAUUSD, BULLISH sur USDJPY.
        """
        curr = surprise_result.get("currency", "USD").upper()
        bias = surprise_result.get("currency_bias", "NEUTRAL")
        sym = symbol.upper()

        if bias == "NEUTRAL":
            return "NEUTRAL"

        # Si la devise est la devise de base (ex: USDJPY pour USD)
        if sym.startswith(curr):
            return "BULLISH" if bias in ("BULLISH", "MILDLY_BULLISH") else "BEARISH"
        # Si la devise est la devise de contrepartie (ex: EURUSD ou XAUUSD pour USD)
        elif sym.endswith(curr) or (curr == "USD" and sym in ("EURUSD", "GBPUSD", "XAUUSD", "AUDUSD", "NZDUSD")):
            return "BEARISH" if bias in ("BULLISH", "MILDLY_BULLISH") else "BULLISH"

        return "NEUTRAL"
