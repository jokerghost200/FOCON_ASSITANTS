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

            # 1. Poids Force relative des devises (30%) - UNIQUEMENT SI LES DONNÉES RÉELLES EXISTENT
            if cs_avail:
                if cs_info["bias"] == "BULLISH":
                    score_bull += 0.30
                    reasons.append(cs_info["reason"])
                elif cs_info["bias"] == "BEARISH":
                    score_bear += 0.30
                    reasons.append(cs_info["reason"])
            else:
                reasons.append("Force devises : " + cs_info.get("reason", "Données MT5 insuffisantes"))

            # 2. Poids Différentiel de taux (25%) - UNIQUEMENT SI LES TAUX SONT CONFIRMÉS FRAIS (<90j)
            if macro_fresh:
                if macro_diff["carry_bias"] == "BULLISH":
                    score_bull += 0.25
                    reasons.append(macro_diff["reason"])
                elif macro_diff["carry_bias"] == "BEARISH":
                    score_bear += 0.25
                    reasons.append(macro_diff["reason"])
            else:
                reasons.append("Carry Trade : " + macro_diff.get("reason", "Taux non vérifiés récemment"))

            # 3. Poids Fondamentaux Macro Officiels (BLS, BCE, Eurostat) & Surprises (25%)
            macro_fund = self._evaluate_macro_fundamentals(base_curr, quote_curr)
            if macro_fund["is_fresh"] and macro_fund["used_indicators"] > 0:
                if macro_fund["bias"] == "BULLISH":
                    score_bull += 0.25
                    reasons.extend(macro_fund["reasons"])
                elif macro_fund["bias"] == "BEARISH":
                    score_bear += 0.25
                    reasons.extend(macro_fund["reasons"])
                else:
                    reasons.extend(macro_fund["reasons"])
            else:
                if macro_fund["reasons"]:
                    reasons.extend(macro_fund["reasons"])
                else:
                    reasons.append(f"Indicateurs officiels : données macro non encore synchronisées pour {base_curr}/{quote_curr}.")

            # 4. Poids Sentiment des actualités (20%) - Uniquement si des actualités bilatérales existent
            if has_news:
                if sent_label == "BULLISH":
                    score_bull += 0.20
                    reasons.append(f"Actualités relatives favorables à {base_curr} vs {quote_curr} ({pair_rel_sentiment:+.2f}).")
                elif sent_label == "BEARISH":
                    score_bear += 0.20
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

            if not macro_fresh or not macro_fund.get("is_fresh", True):
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
                "calendar_confidence": blackout_info.get("calendar_confidence", "APPROXIMATE" if not self.calendar.last_fetch_time else "VERIFIED"),
                "calendar_status": blackout_info.get("calendar_status", "LOCAL_FALLBACK" if not self.calendar.last_fetch_time else "OFFICIAL_LIVE_FEED"),
                "currency_strength": {
                    "available": cs_avail,
                    "base": cs_info.get("base_strength", 50.0),
                    "quote": cs_info.get("quote_strength", 50.0),
                    "differential": cs_info.get("differential", 0.0),
                },
                "interest_rate_differential": macro_diff.get("differential", 0.0),
                "is_rate_stale": not macro_fresh,
                "macro_fundamentals": {
                    "bias": macro_fund["bias"],
                    "score": macro_fund["score"],
                    "is_fresh": macro_fund["is_fresh"],
                    "used_indicators_count": macro_fund["used_indicators"],
                    "reasons": macro_fund["reasons"],
                },
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

    def _evaluate_macro_fundamentals(self, base_curr: str, quote_curr: str) -> Dict[str, Any]:
        """
        Évalue objectivement les fondamentaux macroéconomiques officiels (BLS, BCE, Eurostat) :
        - Taux de chômage relatif (santé de l'emploi)
        - Inflation CPI / IPCH annuelle vs cible de stabilité des prix (2.0%)
        - Croissance du PIB réel YoY
        - Surprises économiques récentes (actual vs forecast) via EventImpactEngine
        
        Retourne :
        - bias : 'BULLISH', 'BEARISH' ou 'NEUTRAL'
        - score : contribution directionnelle relative [-1.0, +1.0]
        - is_fresh : True si les données officielles sont récentes (< 90j) et non corrompues
        - used_indicators : nombre d'indicateurs exploités
        - reasons : explications détaillées
        """
        reasons = []
        macro_score = 0.0
        used_indicators = 0
        is_fresh = True

        # 1. Inflation (CPI / HICP)
        base_cpi = self.memory.get_latest_official_indicator("CPI_YOY", base_curr) or self.memory.get_latest_official_indicator("HICP_YOY", base_curr)
        quote_cpi = self.memory.get_latest_official_indicator("CPI_YOY", quote_curr) or self.memory.get_latest_official_indicator("HICP_YOY", quote_curr)

        if base_cpi and quote_cpi:
            if base_cpi.get("status") in ("UNKNOWN", "STALE") or quote_cpi.get("status") in ("UNKNOWN", "STALE"):
                is_fresh = False
            else:
                try:
                    b_val = float(base_cpi.get("value", 0.0))
                    q_val = float(quote_cpi.get("value", 0.0))
                    cpi_diff = b_val - q_val
                    if cpi_diff >= 0.5:
                        macro_score += 0.35
                        reasons.append(f"Inflation annuelle {base_curr} ({b_val:.1f}%) > {quote_curr} ({q_val:.1f}%) [Pression monétaire plus ferme {base_curr}].")
                    elif cpi_diff <= -0.5:
                        macro_score -= 0.35
                        reasons.append(f"Inflation annuelle {base_curr} ({b_val:.1f}%) < {quote_curr} ({q_val:.1f}%) [Pression monétaire plus ferme {quote_curr}].")
                    used_indicators += 1
                except (ValueError, TypeError):
                    is_fresh = False

        # 2. Chômage (Unemployment) - Règle inversée : un chômage plus bas est signe de vigueur économique
        base_unemp = self.memory.get_latest_official_indicator("UNEMPLOYMENT", base_curr)
        quote_unemp = self.memory.get_latest_official_indicator("UNEMPLOYMENT", quote_curr)

        if base_unemp and quote_unemp:
            if base_unemp.get("status") in ("UNKNOWN", "STALE") or quote_unemp.get("status") in ("UNKNOWN", "STALE"):
                is_fresh = False
            else:
                try:
                    b_u = float(base_unemp.get("value", 0.0))
                    q_u = float(quote_unemp.get("value", 0.0))
                    u_diff = q_u - b_u  # Si q_u > b_u, base a moins de chômage (favorable à base)
                    if u_diff >= 0.5:
                        macro_score += 0.35
                        reasons.append(f"Emploi plus solide : Chômage {base_curr} ({b_u:.1f}%) < {quote_curr} ({q_u:.1f}%).")
                    elif u_diff <= -0.5:
                        macro_score -= 0.35
                        reasons.append(f"Emploi moins solide : Chômage {base_curr} ({b_u:.1f}%) > {quote_curr} ({q_u:.1f}%).")
                    used_indicators += 1
                except (ValueError, TypeError):
                    is_fresh = False

        # 3. Croissance du PIB réel (GDP Growth YoY)
        base_gdp = self.memory.get_latest_official_indicator("GDP_GROWTH_YOY", base_curr)
        quote_gdp = self.memory.get_latest_official_indicator("GDP_GROWTH_YOY", quote_curr)
        if base_gdp and quote_gdp:
            if base_gdp.get("status") in ("UNKNOWN", "STALE") or quote_gdp.get("status") in ("UNKNOWN", "STALE"):
                is_fresh = False
            else:
                try:
                    b_g = float(base_gdp.get("value", 0.0))
                    q_g = float(quote_gdp.get("value", 0.0))
                    g_diff = b_g - q_g
                    if g_diff >= 0.5:
                        macro_score += 0.30
                        reasons.append(f"Croissance PIB {base_curr} ({b_g:.1f}%) > {quote_curr} ({q_g:.1f}%).")
                    elif g_diff <= -0.5:
                        macro_score -= 0.30
                        reasons.append(f"Croissance PIB {base_curr} ({b_g:.1f}%) < {quote_curr} ({q_g:.1f}%).")
                    used_indicators += 1
                except (ValueError, TypeError):
                    is_fresh = False

        # 4. Surprises économiques récentes via EventImpactEngine
        try:
            recent_events = self.memory.get_upcoming_events(currency=base_curr, window_minutes=1440 * 3)
            for ev in recent_events:
                act = ev.get("actual")
                fc = ev.get("forecast")
                if act is not None and fc is not None:
                    surprise = self.event_impact.evaluate_surprise(
                        event_title=ev.get("title", ""),
                        currency=base_curr,
                        actual=act,
                        forecast=fc,
                        previous=ev.get("previous"),
                    )
                    if surprise["currency_bias"] == "BULLISH":
                        macro_score += 0.20
                        reasons.append(f"Surprise éco favorable {base_curr} : {surprise['reason']}")
                    elif surprise["currency_bias"] == "BEARISH":
                        macro_score -= 0.20
                        reasons.append(f"Surprise éco défavorable {base_curr} : {surprise['reason']}")
                    break
        except Exception:
            pass

        if macro_score >= 0.25:
            bias = "BULLISH"
        elif macro_score <= -0.25:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        return {
            "bias": bias,
            "score": round(max(-1.0, min(1.0, macro_score)), 2),
            "is_fresh": is_fresh,
            "used_indicators": used_indicators,
            "reasons": reasons,
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
            "providers_health": {
                "bls": self.bls.get_provider_health(),
                "ecb": self.ecb.get_provider_health(),
                "eurostat": self.eurostat.get_provider_health(),
            },
            "status": "SUCCESS" if total > 0 else "PARTIAL_OR_UNAVAILABLE",
        }

    def actualizar_datos_economicos(self, fetch_official: bool = True, print_table: bool = True) -> Dict[str, Any]:
        """
        Met à jour l'ensemble des flux économiques (Calendrier + Flux RSS + Indicateurs officiels).
        Peut être exécuté périodiquement par le LearningScheduler en arrière-plan.
        Affiche le Tableau de santé économique formaté si print_table=True.
        """
        logger.info("📡 [El Economista] Rafraîchissement des flux économiques gratuits...")
        events = self.calendar.fetch_public_calendar()
        articles = self.news.collect_from_rss_feeds()
        official_summary = {}
        if fetch_official:
            official_summary = self.actualizar_indicadores_oficiales()

        if print_table:
            try:
                self.imprimir_tabla_salud_economica()
            except Exception as _pe:
                logger.debug(f"Erreur affichage tableau santé économique: {_pe}")

        return {
            "events_loaded": len(events),
            "news_articles_loaded": len(articles),
            "official_indicators": official_summary,
            "status": "UPDATED",
        }

    def generar_tabla_salud_economica(self) -> str:
        """
        Génère le Tableau de santé économique complet de FOCON.
        Distingue rigoureusement :
        - Collecte HTTP réussie (connectivité, code HTTP)
        - Données validées (indicateurs parsés et enregistrés)
        - Données fraîches (OFFICIAL_CURRENT vs STALE vs UNKNOWN)
        - Couverture (nombre d'indicateurs et motifs d'échec)
        """
        lines = []
        lines.append("=" * 105)
        lines.append("               TABLEAU DE SANTE ECONOMIQUE -- FOCON ASSISTANT (EL ECONOMISTA)")
        lines.append("=" * 105)

        lines.append("1. FOURNISSEURS OFFICIELS (Statistiques Reelles sans Cle API)")
        lines.append("-" * 105)

        providers = [
            ("BLS (Etats-Unis)", self.bls),
            ("BCE (Zone euro)", self.ecb),
            ("Eurostat (Zone euro)", self.eurostat),
        ]

        all_inds = self.memory.get_official_indicators() if hasattr(self.memory, "get_official_indicators") else []

        for p_label, p_obj in providers:
            p_health = p_obj.get_provider_health() if hasattr(p_obj, "get_provider_health") else {}
            p_name = p_obj.name.upper()
            p_status = p_health.get("status", "UNINITIALIZED")
            last_err = p_health.get("last_error")

            p_inds = [i for i in all_inds if str(i.get("source", "")).upper() == p_name]
            current_cnt = sum(1 for i in p_inds if i.get("status") == "OFFICIAL_CURRENT")
            stale_cnt = sum(1 for i in p_inds if i.get("status") == "STALE")

            if p_status == "UNINITIALIZED":
                display_status = "CACHED_SQLITE" if p_inds else "UNINITIALIZED"
            else:
                display_status = p_status

            summary_tag = "VALIDE & FRAIS" if (p_inds and current_cnt > 0 and not last_err) else ("ERREUR RESEAU" if p_status == "ERROR" else "COUVERTURE PARTIELLE")
            lines.append(f"  * {p_label:<24} : [{display_status}] {len(p_inds)} indicateur(s) | {summary_tag}")
            if last_err:
                lines.append(f"    [!] Anomalie reseau : {last_err}")

            if p_inds:
                for ind in p_inds:
                    name = ind.get("indicator", "UNKNOWN")
                    val = ind.get("value")
                    unit = ind.get("unit", "")
                    period = ind.get("reference_period") or ind.get("effective_date") or "N/A"
                    status = ind.get("status", "UNKNOWN")
                    stat, age_days = self.bls.evaluate_staleness(
                        indicator=name,
                        reference_period=str(period),
                        effective_date=ind.get("effective_date"),
                        published_at=ind.get("published_at"),
                    )
                    icon = "[OK]   " if status == "OFFICIAL_CURRENT" else ("[STALE]" if status == "STALE" else "[ALERT]")
                    val_str = f"{val} {unit}"
                    age_str = f"{age_days:.0f}j" if age_days < 999 else "N/A"
                    lines.append(f"    {icon} {name:<23} : {val_str:<18} | Ref: {str(period):<10} | Age: {age_str:<6} [{status}]")
            else:
                lines.append("    (Aucun indicateur disponible en memoire locale)")
            lines.append("")

        lines.append("2. CALENDRIER ECONOMIQUE & FENETRE DE GEL (News Blackout)")
        lines.append("-" * 105)
        blackout_res = self.calendar.is_in_news_blackout(symbol="EURUSD", window_minutes=30.0)
        cal_prov_status = blackout_res.get("calendar_status", "LOCAL_FALLBACK")
        cal_conf = blackout_res.get("cal_confidence", "APPROXIMATE")
        in_blk = blackout_res.get("in_blackout", False)
        evt_title = blackout_res.get("event_title") or "Aucune annonce majeure imminente"
        mins = blackout_res.get("minutes_to_event", 999.0)

        lines.append(f"  * Source Calendrier    : {cal_prov_status} (Confiance : {cal_conf})")
        lines.append(f"  * Gel News (Blackout)  : {'[BLOQUE] ACTIF (ORDRES MT5 INTERDITS)' if in_blk else '[OK] INACTIF (Trading tolere)'}")
        lines.append(f"  * Prochain Evenement   : {evt_title} (dans {mins:.1f} min)")
        lines.append("")

        lines.append("3. FLUX D'ACTUALITES FINANCIERES & SENTIMENT")
        lines.append("-" * 105)
        news_articles = self.memory.get_recent_news(limit=10) if hasattr(self.memory, "get_recent_news") else []
        lines.append(f"  * Articles en memoire  : {len(news_articles)} article(s) analyse(s)")
        lines.append("=" * 105)
        return "\n".join(lines)

    def imprimir_tabla_salud_economica(self) -> None:
        """Affiche le Tableau de santé économique dans les logs INFO et la console."""
        tabla = self.generar_tabla_salud_economica()
        try:
            print("\n" + tabla + "\n")
        except UnicodeEncodeError:
            import sys
            sys.stdout.buffer.write(("\n" + tabla + "\n").encode("utf-8", errors="replace"))
        logger.info(f"\n{tabla}")
