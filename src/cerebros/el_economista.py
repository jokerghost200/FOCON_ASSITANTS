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
from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from src.economic.economic_memory import EconomicMemory
from src.economic.calendar_provider import CalendarProvider
from src.economic.news_collector import NewsCollector
from src.economic.currency_strength import CurrencyStrengthEngine
from src.economic.macro_analyzer import MacroAnalyzer
from src.economic.event_impact import EventImpactEngine
from src.economic.providers.bls_provider import BLSProvider
from src.economic.providers.ecb_provider import ECBProvider
from src.economic.providers.eurostat_provider import EurostatProvider

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
        bls_provider: Optional[BLSProvider] = None,
        ecb_provider: Optional[ECBProvider] = None,
        eurostat_provider: Optional[EurostatProvider] = None,
        mt5_connector=None,
    ):
        self.memory = memory or EconomicMemory()
        self.calendar = calendar or CalendarProvider(memory=self.memory)
        self.news = news or NewsCollector(memory=self.memory)
        self.currency_strength = currency_strength or CurrencyStrengthEngine(mt5_connector=mt5_connector)
        self.macro = macro or MacroAnalyzer(memory=self.memory)
        self.event_impact = event_impact or EventImpactEngine()
        self.bls = bls_provider or BLSProvider(memory=self.memory)
        self.ecb = ecb_provider or ECBProvider(memory=self.memory)
        self.eurostat = eurostat_provider or EurostatProvider(memory=self.memory)

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
        if sym in ("XAUUSD", "BTCUSD", "ETHUSD"):
            base_curr = sym[:3]
            quote_curr = "USD"
        elif len(sym) >= 6:
            base_curr = sym[:3]
            quote_curr = sym[3:6]
        else:
            base_curr = "USD"
            quote_curr = "USD"

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

            # 4. Sentiment des actualités bilatéral (Base vs Quote)
            sent_base = self.news.get_sentiment_summary(currency=base_curr, hours=24.0)
            sent_quote = self.news.get_sentiment_summary(currency=quote_curr, hours=24.0) if quote_curr != base_curr else {
                "sentiment_score": 0.0,
                "sentiment_label": "NEUTRAL",
                "article_count": 0,
            }

            score_base = float(sent_base.get("sentiment_score", 0.0))
            score_quote = float(sent_quote.get("sentiment_score", 0.0))
            has_news = (sent_base.get("article_count", 0) > 0) or (sent_quote.get("article_count", 0) > 0)

            # Sentiment relatif sur la paire : Base - Quote
            PAIR_SENTIMENT_THRESHOLD = 0.15
            pair_rel_sentiment = round(score_base - score_quote, 2)
            if pair_rel_sentiment >= PAIR_SENTIMENT_THRESHOLD:
                sent_label = "BULLISH"
            elif pair_rel_sentiment <= -PAIR_SENTIMENT_THRESHOLD:
                sent_label = "BEARISH"
            else:
                sent_label = "NEUTRAL"

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

            # Poids Sentiment des actualités (25%) - Uniquement si des actualités bilatérales existent
            if has_news:
                if sent_label == "BULLISH":
                    score_bull += 0.25
                    reasons.append(f"Actualités relatives favorables à {base_curr} vs {quote_curr} ({pair_rel_sentiment:+.2f}).")
                elif sent_label == "BEARISH":
                    score_bear += 0.25
                    reasons.append(f"Actualités relatives défavorables à {base_curr} vs {quote_curr} ({pair_rel_sentiment:+.2f}).")
                else:
                    reasons.append(f"Actualités équilibrées entre {base_curr} ({score_base:+.2f}) et {quote_curr} ({score_quote:+.2f}).")
            else:
                reasons.append(f"Actualités : aucune dépêche récente pour {base_curr}/{quote_curr}.")

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
                conf_str = blackout_info.get("calendar_confidence", "APPROXIMATE")
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

            # Fraîcheur granulaire de chaque source
            now_dt = datetime.now(timezone.utc)
            def _calc_age_seconds(ts_val):
                if not ts_val:
                    return None
                try:
                    if isinstance(ts_val, datetime):
                        dt = ts_val
                    else:
                        dt = datetime.fromisoformat(str(ts_val).replace("Z", "+00:00"))
                    return round((now_dt - dt).total_seconds(), 1)
                except Exception:
                    return None

            sources_freshness = {
                "calendar": {
                    "status": "FRESH" if self.calendar.last_fetch_time else "LOCAL_FALLBACK",
                    "confidence": blackout_info.get("calendar_confidence", "APPROXIMATE"),
                    "last_fetch": self.calendar.last_fetch_time.isoformat() if isinstance(self.calendar.last_fetch_time, datetime) else str(self.calendar.last_fetch_time),
                    "age_seconds": _calc_age_seconds(self.calendar.last_fetch_time),
                },
                "bls": {
                    "status": self.bls.last_status,
                    "last_fetch": self.bls.last_fetch_time,
                    "age_seconds": _calc_age_seconds(self.bls.last_fetch_time),
                },
                "ecb": {
                    "status": self.ecb.last_status,
                    "last_fetch": self.ecb.last_fetch_time,
                    "age_seconds": _calc_age_seconds(self.ecb.last_fetch_time),
                },
                "eurostat": {
                    "status": self.eurostat.last_status,
                    "last_fetch": self.eurostat.last_fetch_time,
                    "age_seconds": _calc_age_seconds(self.eurostat.last_fetch_time),
                },
                "currency_strength_mt5": {
                    "status": "AVAILABLE" if cs_avail else "UNAVAILABLE",
                    "reason": cs_info.get("reason", ""),
                },
            }

            if not macro_fresh:
                overall_freshness = "STALE"
            elif (self.calendar.last_fetch_time or self.bls.last_fetch_time or self.ecb.last_fetch_time):
                overall_freshness = "FRESH"
            else:
                overall_freshness = "LOCAL_FALLBACK"

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
                "official_indicators": {
                    "count": len(self.memory.get_official_indicators(currency_or_region=base_curr)),
                    "latest": self.memory.get_official_indicators(currency_or_region=base_curr, max_age_days=90.0)[:3],
                },
                "news_sentiment": sent_label,
                "news_sentiment_details": {
                    "base_currency": base_curr,
                    "base_score": score_base,
                    "base_articles": sent_base.get("article_count", 0),
                    "quote_currency": quote_curr,
                    "quote_score": score_quote,
                    "quote_articles": sent_quote.get("article_count", 0),
                    "relative_score": pair_rel_sentiment,
                    "pair_sentiment_bias": sent_label,
                },
                "reason": final_reason,
                "recommendation": recommendation,
                "data_freshness": overall_freshness,
                "sources_freshness": sources_freshness,
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

    def actualizar_indicadores_oficiales(self) -> Dict[str, Any]:
        """
        Interroge les connecteurs officiels ouverts (BLS, BCE, Eurostat)
        et stocke les dernières séries dans la mémoire SQLite locale.
        """
        logger.info("🏛️ [El Economista] Collecte des indicateurs officiels (BLS, BCE, Eurostat)...")
        bls_inds = []
        ecb_inds = []
        eurostat_inds = []

        try:
            bls_inds = self.bls.fetch_latest_indicators()
        except Exception as e:
            logger.warning(f"Échec collecte BLS: {e}")

        try:
            ecb_inds = self.ecb.fetch_latest_indicators()
        except Exception as e:
            logger.warning(f"Échec collecte BCE: {e}")

        try:
            eurostat_inds = self.eurostat.fetch_latest_indicators()
        except Exception as e:
            logger.warning(f"Échec collecte Eurostat: {e}")

        total = len(bls_inds) + len(ecb_inds) + len(eurostat_inds)
        return {
            "bls_count": len(bls_inds),
            "ecb_count": len(ecb_inds),
            "eurostat_count": len(eurostat_inds),
            "total_indicators": total,
            "status": "SUCCESS" if total > 0 else "PARTIAL_OR_UNAVAILABLE",
        }

    def actualizar_datos_economicos(self, fetch_official: bool = True) -> Dict[str, Any]:
        """
        Met à jour l'ensemble des flux économiques (Calendrier + Flux RSS + Indicateurs officiels).
        Peut être exécuté périodiquement par le LearningScheduler en arrière-plan.
        """
        logger.info("📡 [El Economista] Rafraîchissement des flux économiques gratuits...")
        events = self.calendar.fetch_public_calendar()
        articles = self.news.collect_from_rss_feeds()
        official_summary = {}
        if fetch_official:
            official_summary = self.actualizar_indicadores_oficiales()

        return {
            "events_loaded": len(events),
            "news_articles_loaded": len(articles),
            "official_indicators": official_summary,
            "status": "UPDATED",
        }
