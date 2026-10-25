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

SÉCURITÉ ET TRAÇABILITÉ DES DONNÉES :
- Chaque taux est associé à sa source officielle, sa date d'effet et sa date de vérification.
- Si un taux dépasse le délai maximal de fraîcheur (TTL 90 jours sans confirmation),
  il est explicitement marqué STALE / PÉRIMÉ et le biais Carry Trade est neutralisé.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta

from src.economic.economic_memory import EconomicMemory

logger = logging.getLogger(__name__)

# TTL maximal de validité d'un taux sans réévaluation officielle
MAX_RATE_AGE_DAYS = 90

# Référentiel initial vérifié des banques centrales avec source et horodatage
CENTRAL_BANK_RATES = {
    "USD": {
        "bank": "Federal Reserve (Fed)",
        "rate": 5.25,
        "inflation": 2.6,
        "stance": "NEUTRAL_HAWKISH",
        "effective_date": "2024-11-07",
        "last_verified_at": "2025-01-15T00:00:00+00:00",
        "source": "Federal Reserve FOMC Statement",
        "description": "Taux élevés, politique restrictive dépendante des données d'emploi et d'inflation.",
    },
    "EUR": {
        "bank": "Banque Centrale Européenne (BCE)",
        "rate": 3.25,
        "inflation": 2.2,
        "stance": "DOVISH",
        "effective_date": "2024-10-17",
        "last_verified_at": "2025-01-15T00:00:00+00:00",
        "source": "ECB Monetary Policy Decisions",
        "description": "Cycle d'assouplissement monétaire pour soutenir la croissance en zone euro.",
    },
    "GBP": {
        "bank": "Bank of England (BoE)",
        "rate": 4.75,
        "inflation": 2.3,
        "stance": "NEUTRAL",
        "effective_date": "2024-11-07",
        "last_verified_at": "2025-01-15T00:00:00+00:00",
        "source": "Bank of England MPC Minutes",
        "description": "Baisse graduelle des taux tout en surveillant la persistance de l'inflation des services.",
    },
    "JPY": {
        "bank": "Bank of Japan (BoJ)",
        "rate": 0.25,
        "inflation": 2.5,
        "stance": "HAWKISH",
        "effective_date": "2024-07-31",
        "last_verified_at": "2025-01-15T00:00:00+00:00",
        "source": "Bank of Japan Outlook Report",
        "description": "Sortie des taux négatifs, hausses graduelles pour normaliser la politique monétaire.",
    },
    "CHF": {
        "bank": "Banque Nationale Suisse (BNS)",
        "rate": 1.00,
        "inflation": 1.1,
        "stance": "DOVISH",
        "effective_date": "2024-09-26",
        "last_verified_at": "2025-01-15T00:00:00+00:00",
        "source": "BNS Monetary Policy Assessment",
        "description": "Inflation très basse, baisses de taux pour freiner l'appréciation excessive du franc suisse.",
    },
    "CAD": {
        "bank": "Bank of Canada (BoC)",
        "rate": 3.75,
        "inflation": 1.9,
        "stance": "DOVISH",
        "effective_date": "2024-10-23",
        "last_verified_at": "2025-01-15T00:00:00+00:00",
        "source": "Bank of Canada Rate Announcement",
        "description": "Assouplissement monétaire vigoureux face au ralentissement économique.",
    },
    "AUD": {
        "bank": "Reserve Bank of Australia (RBA)",
        "rate": 4.35,
        "inflation": 2.8,
        "stance": "HAWKISH",
        "effective_date": "2024-11-05",
        "last_verified_at": "2025-01-15T00:00:00+00:00",
        "source": "RBA Statement on Monetary Policy",
        "description": "Prudence maximale face à une inflation sous-jacente tenace, maintien de taux restrictifs.",
    },
}


class MacroAnalyzer:
    """
    Analyseur des déterminants macroéconomiques fondamentaux.
    Vérifie la fraîcheur des données et refuse d'attribuer un biais carry trade sur des taux périmés.
    """

    def __init__(self, memory: Optional[EconomicMemory] = None, max_age_days: int = MAX_RATE_AGE_DAYS):
        self.memory = memory or EconomicMemory()
        self.max_age_days = max_age_days
        self.central_bank_rates = {k: dict(v) for k, v in CENTRAL_BANK_RATES.items()}
        self._sync_default_rates()

    def _sync_default_rates(self):
        """Enregistre les taux initiaux dans la mémoire SQLite s'ils sont absents."""
        now_str = datetime.now(timezone.utc).isoformat()
        for curr, data in self.central_bank_rates.items():
            existing = self.memory.get_latest_macro_snapshot(curr)
            if not existing:
                self.memory.store_macro_snapshot(
                    currency=curr,
                    interest_rate=data["rate"],
                    inflation_rate=data["inflation"],
                    monetary_stance=data["stance"],
                    updated_at=now_str,
                )

    def get_central_bank_info(self, currency: str) -> Dict[str, Any]:
        """
        Retourne la fiche macroéconomique d'une devise avec son indicateur de fraîcheur.
        """
        curr = currency.upper()
        now = datetime.now(timezone.utc)
        now_str = now.isoformat()

        # Actifs non adossés à des banques centrales souveraines (ex. Métaux XAU/XAG, Crypto BTC/ETH, Matières premières)
        if curr not in self.central_bank_rates:
            mem_snap = self.memory.get_latest_macro_snapshot(curr)
            if mem_snap:
                rate = float(mem_snap.get("interest_rate", 0.0))
                inflation = float(mem_snap.get("inflation_rate", 0.0))
                stance = str(mem_snap.get("monetary_stance", "NEUTRAL"))
                updated_at = mem_snap.get("updated_at") or now_str
                try:
                    age_days = (now - datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))).total_seconds() / 86400.0
                    is_stale = age_days > self.max_age_days
                except Exception:
                    is_stale = False
                    age_days = 0.0
                return {
                    "bank": f"Custom Asset ({curr})",
                    "rate": rate,
                    "inflation": inflation,
                    "stance": stance,
                    "effective_date": "N/A",
                    "last_verified_at": updated_at,
                    "source": "DATABASE_SNAPSHOT",
                    "description": f"Données macro personnalisées pour {curr}.",
                    "is_stale": is_stale,
                    "age_days": round(age_days, 1),
                    "freshness": "STALE" if is_stale else "FRESH",
                    "is_sovereign": False,
                }
            return {
                "bank": f"Non-Sovereign Asset ({curr})",
                "rate": 0.0,
                "inflation": 0.0,
                "stance": "NEUTRAL",
                "effective_date": "N/A",
                "last_verified_at": now_str,
                "source": "NON_SOVEREIGN_COMMODITY",
                "description": f"Actif non souverain ({curr}) sans taux de banque centrale applicable.",
                "is_stale": False,
                "age_days": 0.0,
                "freshness": "FRESH",
                "is_sovereign": False,
            }

        mem_snap = self.memory.get_latest_macro_snapshot(curr)
        ref_data = self.central_bank_rates[curr]
        info = dict(ref_data)
        if mem_snap:
            info["rate"] = float(mem_snap.get("interest_rate", info["rate"]))
            info["inflation"] = float(mem_snap.get("inflation_rate", info["inflation"]))
            info["stance"] = str(mem_snap.get("monetary_stance", info["stance"]))
            if "updated_at" in mem_snap and mem_snap["updated_at"]:
                info["last_verified_at"] = mem_snap["updated_at"]

        # Calcul de la fraîcheur (staleness) pour les banques centrales souveraines
        is_stale = False
        age_days = 0.0
        try:
            verified_time = datetime.fromisoformat(str(info["last_verified_at"]).replace("Z", "+00:00"))
            age_days = (now - verified_time).total_seconds() / 86400.0
            if age_days > self.max_age_days:
                is_stale = True
        except Exception:
            is_stale = True

        info["is_stale"] = is_stale
        info["age_days"] = round(age_days, 1)
        info["freshness"] = "STALE" if is_stale else "FRESH"
        info["is_sovereign"] = True
        return info

    def calculate_interest_differential(self, symbol: str) -> Dict[str, Any]:
        """
        Calcule le différentiel de taux d'intérêt (Carry Trade) pour une paire.
        Neutralise le biais si l'une des devises repose sur des données périmées.
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

        # ── Contrôle de Fraîcheur des Données ──
        if base_info.get("is_stale") or quote_info.get("is_stale"):
            stale_curr = base if base_info.get("is_stale") else quote
            logger.warning(
                f"⚠️ [Carry Trade] Taux {stale_curr} périmé (non vérifié depuis > {self.max_age_days}j). "
                f"Biais de taux neutralisé par précaution."
            )
            return {
                "symbol": sym,
                "base_currency": base,
                "quote_currency": quote,
                "base_rate": rate_base,
                "quote_rate": rate_quote,
                "differential": diff,
                "carry_bias": "NEUTRAL",
                "is_stale": True,
                "freshness": "STALE",
                "reason": (
                    f"Taux directeurs {base} ou {quote} non vérifiés récemment "
                    f"({stale_curr} périmé > {self.max_age_days} jours). Biais Carry Trade neutralisé par prudence."
                ),
            }

        # Détermination du biais Carry Trade uniquement si les données sont confirmées fraîches
        if diff >= 1.50:
            carry_bias = "BULLISH"
            reason = f"Différentiel de taux favorable (+{diff:.2f}%) à {base} ({rate_base}%) vs {quote} ({rate_quote}%)."
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
            "is_stale": False,
            "freshness": "FRESH",
            "base_stance": base_info.get("stance"),
            "quote_stance": quote_info.get("stance"),
            "source_base": base_info.get("source"),
            "source_quote": quote_info.get("source"),
            "reason": reason,
        }
