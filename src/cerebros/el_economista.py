# -*- coding: utf-8 -*-
"""
CEREBRO 6 : EL ECONOMISTA (Cerveau Macroéconomique & Fondamental)
================================================================
Nom Espagnol : El Economista ("L'Économiste")
Rôle : Comprend la macroéconomie, surveille le calendrier économique, les taux directeurs,
la force relative des devises (CSI) et le sentiment des actualités financières pour guider
El Observador et El Estratega, tout en imposant un gel strict (News Blackout) en cas d'annonce critique.

Composants fédérés :
- CalendarProvider : Détection des publications économiques majeures (NFP, CPI, Taux)
- NewsCollector : Collecte d'actualités financières gratuites et analyse de sentiment locale
- CurrencyStrengthEngine : Force relative des devises (CSI 0-100) calculée depuis MT5
- MacroAnalyzer : Taux directeurs, inflation et différentiels de taux (Carry Trade)
- EventImpactEngine : Écart aux prévisions et surprise économique
- EconomicMemory : Mémoire locale SQLite des événements et des réactions de marché
"""

import logging
from typing import Dict, Any, Optional

from src.economic.economic_memory import EconomicMemory
from src.economic.calendar_provider import CalendarProvider
from src.economic.news_collector import NewsCollector
from src.economic.currency_strength import CurrencyStrengthEngine
from src.economic.macro_analyzer import MacroAnalyzer
from src.economic.event_impact import EventImpactEngine

logger = logging.getLogger(__name__)


class ElEconomista:
    """
    Cerveau Macroéconomique & Fondamental (El Economista)
    Fournit la couche d'intelligence économique à l'architecture FATI.
    """

    def __init__(
        self,
        memory: Optional[EconomicMemory] = None,
        calendar: Optional[CalendarProvider] = None,
        news: Optional[NewsCollector] = None,
        currency_strength: Optional[CurrencyStrengthEngine] = None,
        macro: Optional[MacroAnalyzer] = None,
        event_impact: Optional[EventImpactEngine] = None,
        mt5_connector=None,
    ):
        self.memory = memory or EconomicMemory()
        self.calendar = calendar or CalendarProvider(memory=self.memory)
        self.news = news or NewsCollector(memory=self.memory)
        self.currency_strength = currency_strength or CurrencyStrengthEngine(mt5_connector=mt5_connector)
        self.macro = macro or MacroAnalyzer(memory=self.memory)
        self.event_impact = event_impact or EventImpactEngine()

        # Initialisation du calendrier en arrière-plan
        try:
            self.calendar.fetch_public_calendar()
        except Exception as e:
            logger.debug(f"Initialisation calendrier ElEconomista: {e}")

    def analizar_contexto_economico(
        self,
        symbol: str,
        news_blackout_minutes: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Génère une analyse macroéconomique globale et structurée pour un symbole donné :
        1. Vérification de la fenêtre de gel des annonces (News Blackout)
        2. Force relative de la devise de base vs devise de cotation (CSI)
        3. Différentiel de taux des banques centrales (Carry Trade)
        4. Sentiment récent des actualités
        """
        sym = symbol.upper()
        curr = "USD" if "USD" in sym or sym in ("XAUUSD", "BTCUSD", "ETHUSD") else sym[:3]

        try:
            # 1. Vérification du calendrier et gel d'annonces
            blackout_info = self.calendar.is_in_news_blackout(
                symbol=sym, window_minutes=news_blackout_minutes, min_impact="HIGH"
            )
            is_blackout = bool(blackout_info.get("in_blackout", False))

            # 2. Force relative des devises (MT5 CSI) - vérification de disponibilité
            cs_info = self.currency_strength.get_pair_bias(sym)
            cs_avail = bool(cs_info.get("available", True) and cs_info.get("status") != "UNAVAILABLE")

            # 3. Différentiel de taux directeurs - vérification de fraîcheur
            macro_diff = self.macro.calculate_interest_differential(sym)
            macro_fresh = not bool(macro_diff.get("is_stale", False))

            # 4. Sentiment des actualités
            sentiment_info = self.news.get_sentiment_summary(currency=curr, hours=24.0)

            # ── Synthèse Macroéconomique Globale ──
            score_bull = 0.0
            score_bear = 0.0
            reasons = []

            # Poids Force relative des devises (40%) - UNIQUEMENT SI LES DONNÉES RÉELLES EXISTENT
            if cs_avail:
                if cs_info["bias"] == "BULLISH":
                    score_bull += 0.40
                    reasons.append(cs_info["reason"])
                elif cs_info["bias"] == "BEARISH":
                    score_bear += 0.40
                    reasons.append(cs_info["reason"])
            else:
                reasons.append("Force devises : " + cs_info.get("reason", "Données MT5 insuffisantes"))

            # Poids Différentiel de taux (35%) - UNIQUEMENT SI LES TAUX SONT CONFIRMÉS FRAIS (<90j)
            if macro_fresh:
                if macro_diff["carry_bias"] == "BULLISH":
                    score_bull += 0.35
                    reasons.append(macro_diff["reason"])
                elif macro_diff["carry_bias"] == "BEARISH":
                    score_bear += 0.35
                    reasons.append(macro_diff["reason"])
            else:
                reasons.append("Carry Trade : " + macro_diff.get("reason", "Taux non vérifiés récemment"))

            # Poids Sentiment des actualités (25%) - Uniquement si des actualités ont été trouvées
            sent_label = sentiment_info.get("sentiment_label", "NEUTRAL")
            if sent_label == "BULLISH":
                score_bull += 0.25
                reasons.append(f"Actualités favorables ({sentiment_info.get('sentiment_score', 0):+.2f}).")
            elif sent_label == "BEARISH":
                score_bear += 0.25
                reasons.append(f"Actualités défavorables ({sentiment_info.get('sentiment_score', 0):+.2f}).")

            # Biais résultant
            if score_bull >= score_bear + 0.15:
                overall_bias = "BULLISH"
                confidence = round(score_bull, 2)
            elif score_bear >= score_bull + 0.15:
                overall_bias = "BEARISH"
                confidence = round(score_bear, 2)
            else:
                overall_bias = "NEUTRAL"
                confidence = 0.50

            # Risque d'événement
            if is_blackout:
                event_risk = "CRITICAL" if blackout_info.get("impact") == "CRITICAL" else "HIGH"
                recommendation = "NO_TRADE_NEWS"
                conf_str = blackout_info.get("calendar_confidence", "VERIFIED")
                final_reason = f"🛑 [GEL ANNONCE ÉCONOMIQUE] {blackout_info['reason']}"
            else:
                event_risk = "LOW"
                if overall_bias == "BULLISH":
                    recommendation = "FAVOR_BUY"
                    final_reason = f"Contexte macro favorable à l'achat : {' | '.join(reasons)}"
                elif overall_bias == "BEARISH":
                    recommendation = "FAVOR_SELL"
                    final_reason = f"Contexte macro favorable à la vente : {' | '.join(reasons)}"
                else:
                    recommendation = "NORMAL"
                    final_reason = f"Contexte macro neutre ou équilibré ({' | '.join(reasons)})"

            return {
                "cerebro_origen": "El Economista",
                "symbol": sym,
                "status": "OK",
                "bias": overall_bias,
                "confidence": confidence,
                "event_risk": event_risk,
                "high_impact_window": is_blackout,
                "minutes_to_next_event": blackout_info.get("minutes_to_event", 999.0),
                "next_event_title": blackout_info.get("event_title"),
                "calendar_confidence": blackout_info.get("calendar_confidence", "VERIFIED"),
                "currency_strength": {
                    "available": cs_avail,
                    "base": cs_info.get("base_strength", 50.0),
                    "quote": cs_info.get("quote_strength", 50.0),
                    "differential": cs_info.get("differential", 0.0),
                },
                "interest_rate_differential": macro_diff.get("differential", 0.0),
                "is_rate_stale": not macro_fresh,
                "news_sentiment": sentiment_info.get("sentiment_label", "NEUTRAL"),
                "reason": final_reason,
                "recommendation": recommendation,
                "data_freshness": "FRESH" if self.calendar.last_fetch_time else "LOCAL_FALLBACK",
            }
        except Exception as e:
            logger.error(f"🛑 [El Economista] Erreur critique analyse macro pour {symbol}: {e}")
            return {
                "cerebro_origen": "El Economista",
                "symbol": sym,
                "status": "ERROR",
                "bias": "NEUTRAL",
                "confidence": 0.0,
                "event_risk": "CRITICAL",
                "high_impact_window": True,
                "recommendation": "NO_TRADE_NEWS",
                "reason": f"Panne critique du cerveau économique ({e}) : Sécurité Fail-Closed activée.",
                "data_freshness": "ERROR",
            }

    def es_periodo_bloqueo_noticias(self, symbol: str, window_minutes: float = 30.0) -> bool:
        """
        Vérification rapide : indique si le symbole est actuellement en période de gel de nouvelles.
        """
        res = self.calendar.is_in_news_blackout(symbol=symbol, window_minutes=window_minutes, min_impact="HIGH")
        return bool(res.get("in_blackout", False))

    def actualizar_datos_economicos(self) -> Dict[str, Any]:
        """
        Met à jour l'ensemble des flux économiques (Calendrier + Flux RSS).
        Peut être exécuté périodiquement par le LearningScheduler.
        """
        logger.info("📡 [El Economista] Rafraîchissement des flux économiques gratuits...")
        events = self.calendar.fetch_public_calendar()
        articles = self.news.collect_from_rss_feeds()
        return {
            "events_loaded": len(events),
            "news_articles_loaded": len(articles),
            "status": "UPDATED",
        }
