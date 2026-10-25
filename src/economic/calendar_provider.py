# -*- coding: utf-8 -*-
"""
FATI Calendar Provider (Fournisseur de Calendrier Économique Gratuit)
====================================================================
Fournit le flux des événements économiques majeurs sans nécessiter de clé API :
- Téléchargement des calendriers publics gratuits (ex: FinanceCalendar)
- Générateur déterministe de récurrence (NFP, CPI, FOMC, BCE, BoE) comme filet de sécurité 100% hors-ligne
- Détection immédiate des fenêtres de gel des ordres ("News Blackout Window")
- Intégration directe avec EconomicMemory (SQLite)
"""

import json
import logging
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta

from src.economic.economic_memory import EconomicMemory

logger = logging.getLogger(__name__)

# Liste des devises majeures analysées par FOCON
CURRENCY_MAP = {
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"),
    "USDCHF": ("USD", "CHF"),
    "USDCAD": ("USD", "CAD"),
    "AUDUSD": ("AUD", "USD"),
    "NZDUSD": ("NZD", "USD"),
    "XAUUSD": ("USD", "USD"),  # L'or réagit violemment aux annonces du Dollar
    "BTCUSD": ("USD", "USD"),
    "ETHUSD": ("USD", "USD"),
}


class CalendarProvider:
    """
    Fournisseur autonome du calendrier économique avec tolérance totale aux pannes.
    """

    def __init__(self, memory: Optional[EconomicMemory] = None, timeout_seconds: float = 5.0):
        self.memory = memory or EconomicMemory()
        self.timeout_seconds = timeout_seconds
        self.last_fetch_time: Optional[datetime] = None

    def fetch_public_calendar(self) -> List[Dict[str, Any]]:
        """
        Interroge un flux de calendrier économique public gratuit (sans clé API).
        Si la connexion échoue, active automatiquement le fallback déterministe local.
        """
        events: List[Dict[str, Any]] = []
        
        # Endpoint public gratuit (Finance Calendar ou miroir ouvert)
        url = "https://www.financecalendar.com/api/forex-calendar/"
        headers = {
            "User-Agent": "FoconAssistant/1.0 (Public Economic Analysis)",
            "Accept": "application/json",
        }

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                if response.status == 200:
                    raw_data = response.read().decode("utf-8")
                    data = json.loads(raw_data)
                    raw_events = data.get("events") if isinstance(data, dict) else (data if isinstance(data, list) else [])
                    
                    for ev in raw_events:
                        parsed = self._normalize_event(ev)
                        if parsed:
                            events.append(parsed)
                    
                    if events:
                        self.memory.store_events_bulk(events)
                        self.last_fetch_time = datetime.now(timezone.utc)
                        logger.info(f"📅 [El Economista] {len(events)} événements économiques récupérés via FinanceCalendar.")
                        return events
        except Exception as e:
            logger.warning(f"Calendrier public inaccessible ({e}). Activation du calendrier récurrent local.")

        # Fallback hors-ligne garanti
        events = self.generate_recurring_fallback_events()
        self.memory.store_events_bulk(events)
        return events

    def _normalize_event(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalise la structure d'un événement externe vers le schéma standard FATI."""
        try:
            title = str(raw.get("title") or raw.get("name") or "")
            if not title:
                return None
            
            curr = str(raw.get("currency") or raw.get("country") or "USD").upper()
            impact_raw = str(raw.get("impact") or raw.get("importance") or "MEDIUM").upper()
            
            if impact_raw in ("3", "HIGH", "CRITICAL", "RED"):
                impact = "HIGH"
            elif impact_raw in ("2", "MEDIUM", "ORANGE"):
                impact = "MEDIUM"
            else:
                impact = "LOW"

            # Parser le temps
            time_raw = raw.get("date") or raw.get("time") or raw.get("datetime")
            event_time = datetime.now(timezone.utc).isoformat()
            if time_raw:
                try:
                    if isinstance(time_raw, str):
                        event_time = datetime.fromisoformat(time_raw.replace("Z", "+00:00")).isoformat()
                except Exception:
                    pass

            return {
                "title": title,
                "currency": curr,
                "country": str(raw.get("country", curr)),
                "impact": impact,
                "event_time": event_time,
                "actual": self._to_float(raw.get("actual")),
                "forecast": self._to_float(raw.get("forecast")),
                "previous": self._to_float(raw.get("previous")),
                "source": "FINANCE_CALENDAR_PUBLIC",
            }
        except Exception:
            return None

    @staticmethod
    def _to_float(val: Any) -> Optional[float]:
        if val is None or val == "":
            return None
        try:
            clean = str(val).replace("%", "").replace("K", "").replace("M", "").replace(",", "").strip()
            return float(clean)
        except (ValueError, TypeError):
            return None

    def generate_recurring_fallback_events(self) -> List[Dict[str, Any]]:
        """
        Génère les fenêtres critiques des publications macroéconomiques majeures
        (NFP, CPI, Décisions Taux) de façon déterministe pour éviter d'être aveugle hors-ligne.
        """
        now = datetime.now(timezone.utc)
        events = []

        # 1. US Non-Farm Payrolls (Premier Vendredi du mois à 12h30 UTC en été / 13h30 UTC en hiver)
        year, month = now.year, now.month
        first_day = datetime(year, month, 1, 12, 30, tzinfo=timezone.utc)
        first_friday_day = 1 + (4 - first_day.weekday()) % 7
        nfp_date = datetime(year, month, first_friday_day, 12, 30, tzinfo=timezone.utc)
        
        events.append({
            "id": f"nfp_{year}_{month}",
            "title": "US Non-Farm Payrolls & Unemployment Rate (NFP)",
            "country": "United States",
            "currency": "USD",
            "impact": "CRITICAL",
            "event_time": nfp_date.isoformat(),
            "source": "DETERMINISTIC_SCHEDULE",
        })

        # 2. US Consumer Price Index (CPI) - Vers le 12 du mois à 12h30 UTC
        cpi_date = datetime(year, month, 12, 12, 30, tzinfo=timezone.utc)
        events.append({
            "id": f"cpi_us_{year}_{month}",
            "title": "US Consumer Price Index (CPI MoM/YoY)",
            "country": "United States",
            "currency": "USD",
            "impact": "CRITICAL",
            "event_time": cpi_date.isoformat(),
            "source": "DETERMINISTIC_SCHEDULE",
        })

        # 3. FOMC Interest Rate Decision (Mercredis clés vers 18h00 UTC)
        fomc_date = datetime(year, month, 18, 18, 0, tzinfo=timezone.utc)
        events.append({
            "id": f"fomc_{year}_{month}",
            "title": "Fed Interest Rate Decision & FOMC Press Conference",
            "country": "United States",
            "currency": "USD",
            "impact": "CRITICAL",
            "event_time": fomc_date.isoformat(),
            "source": "DETERMINISTIC_SCHEDULE",
        })

        # 4. ECB Interest Rate Decision (Jeudis clés vers 12h15 UTC)
        ecb_date = datetime(year, month, 22, 12, 15, tzinfo=timezone.utc)
        events.append({
            "id": f"ecb_{year}_{month}",
            "title": "ECB Monetary Policy Statement & Press Conference",
            "country": "Eurozone",
            "currency": "EUR",
            "impact": "HIGH",
            "event_time": ecb_date.isoformat(),
            "source": "DETERMINISTIC_SCHEDULE",
        })

        return events

    def is_in_news_blackout(
        self,
        symbol: str,
        window_minutes: float = 30.0,
        min_impact: str = "HIGH",
    ) -> Dict[str, Any]:
        """
        Vérifie si le symbole se trouve actuellement dans une fenêtre d'annonce majeure.
        Période protégée : [-window_minutes avant l'annonce, +window_minutes après l'annonce].
        """
        sym = symbol.upper()
        if sym in CURRENCY_MAP:
            base_curr, quote_curr = CURRENCY_MAP[sym]
        elif len(sym) >= 6:
            base_curr, quote_curr = sym[:3], sym[3:6]
        else:
            base_curr, quote_curr = "USD", "USD"

        # Recherche des événements pertinents dans la fenêtre
        events_base = self.memory.get_upcoming_events(
            currency=base_curr, window_minutes=window_minutes, min_impact=min_impact
        )
        events_quote = self.memory.get_upcoming_events(
            currency=quote_curr, window_minutes=window_minutes, min_impact=min_impact
        ) if quote_curr != base_curr else []

        all_events = events_base + events_quote
        now = datetime.now(timezone.utc)

        for ev in all_events:
            try:
                ev_time = datetime.fromisoformat(ev["event_time"].replace("Z", "+00:00"))
                delta_minutes = (ev_time - now).total_seconds() / 60.0
                
                # Si l'événement est entre -window_minutes (passé récent) et +window_minutes (futur proche)
                if -window_minutes <= delta_minutes <= window_minutes:
                    return {
                        "in_blackout": True,
                        "minutes_to_event": round(delta_minutes, 1),
                        "event_title": ev.get("title"),
                        "currency": ev.get("currency"),
                        "impact": ev.get("impact"),
                        "reason": f"Publication majeure à proximité ({ev.get('title')} dans {delta_minutes:+.0f} min).",
                    }
            except Exception:
                continue

        # Aucun événement à risque immédiat
        return {
            "in_blackout": False,
            "minutes_to_event": 999.0,
            "event_title": None,
            "currency": None,
            "impact": "NONE",
            "reason": "Aucune publication à fort impact dans la fenêtre.",
        }
