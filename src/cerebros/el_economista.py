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

            # 5. Radar Mondial Inter-Marchés (Risk-On/Risk-Off, Pétrole -> CAD, Chine -> AUD/NZD, Sentiment Crypto 24/7)
            world_info = self.analizar_mercados_mundiales(sym, base_curr, quote_curr)
            if world_info["bull_bonus"] > 0:
                score_bull += world_info["bull_bonus"]
                reasons.extend(world_info["reasons"])
            elif world_info["bear_bonus"] > 0:
                score_bear += world_info["bear_bonus"]
                reasons.extend(world_info["reasons"])
            elif world_info["reasons"]:
                reasons.extend(world_info["reasons"])

            # Biais résultant
            if score_bull >= score_bear + 0.15:
                overall_bias = "BULLISH"
                confidence = round(min(1.0, score_bull), 2)
            elif score_bear >= score_bull + 0.15:
                overall_bias = "BEARISH"
                confidence = round(min(1.0, score_bear), 2)
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
        base_cpi = (
            self.memory.get_latest_official_indicator("CPI_YOY", base_curr)
            or self.memory.get_latest_official_indicator("HICP_INFLATION", base_curr)
            or self.memory.get_latest_official_indicator("HICP_YOY", base_curr)
        )
        quote_cpi = (
            self.memory.get_latest_official_indicator("CPI_YOY", quote_curr)
            or self.memory.get_latest_official_indicator("HICP_INFLATION", quote_curr)
            or self.memory.get_latest_official_indicator("HICP_YOY", quote_curr)
        )

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

    def analizar_mercados_mundiales(
        self,
        symbol: str,
        base_curr: str,
        quote_curr: str,
    ) -> Dict[str, Any]:
        """
        Radar Inter-Marchés Mondial :
        1. Climat de Risque Mondial (RISK-OFF vs RISK-ON) :
           - En RISK-OFF (tensions géopolitiques, krachs, crises), renforce les Valeurs Refuges (XAU, JPY, CHF, USD)
             et pénalise les actifs à bêta élevé (AUD, NZD, GBP, CAD).
           - En RISK-ON (optimisme mondial), renforce les devises de croissance/commodities (AUD, NZD, CAD, EUR) et la Crypto.
        2. Corrélation Énergie / Pétrole (OIL -> CAD) :
           - Le Dollar Canadien (CAD) est un pétrodollar. Un sentiment pétrolier haussier renforce CAD
             (haussier sur CADJPY, baissier sur USDCAD/EURCAD).
        3. Sentiment Mondial Crypto 24/7 (BTC/ETH/SOL/XRP/BNB) :
           - Exploite les flux spécialisés Crypto pour orienter les paires Crypto la semaine et le week-end.
        """
        bull_bonus = 0.0
        bear_bonus = 0.0
        reasons: List[str] = []

        safe_havens = {"XAU", "XAG", "JPY", "CHF", "USD"}
        risk_currencies = {"AUD", "NZD", "CAD", "GBP", "EUR"}
        crypto_bases = {"BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT", "LTC"}

        # 1. Climat de risque mondial (Risk-On / Risk-Off)
        risk_regime_info = {"regime": "NEUTRAL", "score": 0.0, "label": ""}
        if hasattr(self.news, "get_global_risk_regime"):
            try:
                risk_regime_info = self.news.get_global_risk_regime(hours=24.0)
            except Exception:
                pass

        g_regime = risk_regime_info.get("regime", "NEUTRAL")
        if g_regime == "RISK_OFF":
            if base_curr in safe_havens and quote_curr in risk_currencies:
                bull_bonus += 0.20
                reasons.append(f"Climat Mondial RISK-OFF : valeur refuge {base_curr} recherchee face a {quote_curr}.")
            elif base_curr in risk_currencies and quote_curr in safe_havens:
                bear_bonus += 0.20
                reasons.append(f"Climat Mondial RISK-OFF : fuite vers la valeur refuge {quote_curr} au detriment de {base_curr}.")
            elif base_curr == "XAU":
                bull_bonus += 0.22
                reasons.append("Climat Mondial RISK-OFF : forte demande refuge sur l'Or (XAU).")
        elif g_regime == "RISK_ON":
            if (base_curr in risk_currencies or base_curr in crypto_bases) and quote_curr in safe_havens:
                bull_bonus += 0.18
                reasons.append(f"Climat Mondial RISK-ON : appetit mondial favorable a {base_curr} face a {quote_curr}.")
            elif base_curr in safe_havens and quote_curr in risk_currencies and base_curr != "XAU":
                bear_bonus += 0.18
                reasons.append(f"Climat Mondial RISK-ON : rotation vers la devise de croissance {quote_curr}.")

        # 2. Corrélation Pétrole Mondial -> Dollar Canadien (CAD)
        if base_curr == "CAD" or quote_curr == "CAD":
            try:
                oil_sent = self.news.get_sentiment_summary(currency="OIL", hours=24.0)
                if oil_sent.get("article_count", 0) > 0:
                    oil_lbl = oil_sent.get("sentiment_label", "NEUTRAL")
                    oil_sc = float(oil_sent.get("sentiment_score", 0.0))
                    if oil_lbl == "BULLISH":
                        if base_curr == "CAD":
                            bull_bonus += 0.18
                            reasons.append(f"Correlation Petrole/CAD : hausse du petrole ({oil_sc:+.2f}) soutient le CAD.")
                        else:  # ex: USDCAD, EURCAD
                            bear_bonus += 0.18
                            reasons.append(f"Correlation Petrole/CAD : hausse du petrole ({oil_sc:+.2f}) renforce CAD (pression baissiere sur {symbol}).")
                    elif oil_lbl == "BEARISH":
                        if base_curr == "CAD":
                            bear_bonus += 0.18
                            reasons.append(f"Correlation Petrole/CAD : baisse du petrole ({oil_sc:+.2f}) affaiblit le CAD.")
                        else:  # ex: USDCAD
                            bull_bonus += 0.18
                            reasons.append(f"Correlation Petrole/CAD : baisse du petrole ({oil_sc:+.2f}) affaiblit CAD (soutien haussier sur {symbol}).")
            except Exception:
                pass

        # 3. Sentiment Mondial Crypto 24/7 (pour BTCUSD, ETHUSD, SOLUSD, etc.)
        if base_curr in crypto_bases or any(c in symbol.upper() for c in crypto_bases):
            try:
                crypto_sent = self.news.get_sentiment_summary(currency="BTC", hours=24.0)
                if crypto_sent.get("article_count", 0) > 0:
                    c_lbl = crypto_sent.get("sentiment_label", "NEUTRAL")
                    c_sc = float(crypto_sent.get("sentiment_score", 0.0))
                    if c_lbl == "BULLISH":
                        bull_bonus += 0.25
                        reasons.append(f"Sentiment Crypto Mondial HAUSSIER ({c_sc:+.2f} sur {crypto_sent.get('article_count')} articles).")
                    elif c_lbl == "BEARISH":
                        bear_bonus += 0.25
                        reasons.append(f"Sentiment Crypto Mondial BAISSIER ({c_sc:+.2f} sur {crypto_sent.get('article_count')} articles).")
            except Exception:
                pass

        return {
            "global_risk_regime": g_regime,
            "bull_bonus": round(bull_bonus, 2),
            "bear_bonus": round(bear_bonus, 2),
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
        Génère le Tableau de santé économique complet et certifié de FOCON.
        Distingue rigoureusement :
        - Diagnostic réseau & HTTP (succès/échecs, dernier code HTTP, date de collecte)
        - Validation métier (parsing et typage des séries attendues)
        - Fraîcheur économique réelle (statut en BD, statut recalculé, âge en jours)
        - Couverture par fournisseur (CURRENT, STALE, UNKNOWN, manquants)
        - Couverture temporelle et niveau de confiance du calendrier économique
        """
        lines = []
        lines.append("=" * 115)
        lines.append("               TABLEAU DE SANTE ECONOMIQUE -- FOCON ASSISTANT (EL ECONOMISTA)")
        lines.append("=" * 115)

        lines.append("1. FOURNISSEURS OFFICIELS (Statistiques Reelles sans Cle API)")
        lines.append("-" * 115)

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
            last_http_code = p_health.get("last_http_code")
            last_fetch = p_health.get("last_fetch_time")
            expected_series = getattr(p_obj, "expected_indicators", [])

            p_inds = [i for i in all_inds if str(i.get("source", "")).upper() == p_name]
            
            # Regrouper par indicateur pour évaluer la fraîcheur sur la dernière observation de chaque série
            latest_by_indicator: Dict[str, Dict[str, Any]] = {}
            for ind in p_inds:
                iname = str(ind.get("indicator", "UNKNOWN")).upper()
                curr_item = latest_by_indicator.get(iname)
                if not curr_item:
                    latest_by_indicator[iname] = ind
                else:
                    d_curr = curr_item.get("effective_date") or curr_item.get("published_at") or curr_item.get("reference_period") or ""
                    d_new = ind.get("effective_date") or ind.get("published_at") or ind.get("reference_period") or ""
                    if str(d_new) > str(d_curr):
                        latest_by_indicator[iname] = ind

            active_inds = list(latest_by_indicator.values())

            # Recalcul de fraîcheur indépendant et décompte strict sur les observations actives les plus récentes
            recomputed_current = 0
            recomputed_stale = 0
            recomputed_unknown = 0
            validated_count = p_health.get("last_validated_indicators_count", len(active_inds))

            for ind in active_inds:
                name = ind.get("indicator", "UNKNOWN")
                period = ind.get("reference_period")
                eff_date = ind.get("effective_date")
                pub_date = ind.get("published_at")
                
                # Calcul de fraîcheur universel et indépendant
                stat_recalc, _ = p_obj.evaluate_staleness(
                    indicator=name,
                    reference_period=period,
                    effective_date=eff_date,
                    published_at=pub_date,
                )
                if stat_recalc == "OFFICIAL_CURRENT":
                    recomputed_current += 1
                elif stat_recalc == "STALE":
                    recomputed_stale += 1
                else:
                    recomputed_unknown += 1

            total_series_available = len(active_inds)
            missing_count = max(0, len(expected_series) - total_series_available)

            if p_status == "UNINITIALIZED":
                display_status = "CACHED_SQLITE" if p_inds else "UNINITIALIZED"
            else:
                display_status = p_status

            # Règle stricte d'étiquetage : interdiction absolue de "VALIDE & FRAIS" si partiel, périmé ou en erreur
            if p_status == "ERROR" or (last_err and total_series_available == 0):
                summary_tag = "ERREUR RESEAU / HTTP"
            elif missing_count > 0 or (expected_series and validated_count < len(expected_series)):
                summary_tag = f"COUVERTURE PARTIELLE ({total_series_available}/{len(expected_series)})"
            elif recomputed_stale > 0:
                summary_tag = f"DONNEES PERIMEES ({recomputed_stale} STALE)"
            elif recomputed_unknown > 0:
                summary_tag = f"STATUT INCERTAIN ({recomputed_unknown} UNKNOWN)"
            elif total_series_available > 0 and recomputed_current == total_series_available and not last_err:
                summary_tag = "VALIDE & FRAIS (COMPLET)"
            else:
                summary_tag = "COUVERTURE PARTIELLE"

            lines.append(f"  * {p_label:<24} : [{display_status}] {total_series_available} série(s) | {summary_tag}")
            
            # Affichage détaillé des métriques du fournisseur
            fetch_str = str(last_fetch)[:19] if last_fetch else "Aucune"
            code_str = str(last_http_code) if last_http_code else "N/A"
            lines.append(
                f"    [Bilan] Validés: {validated_count} | CURRENT: {recomputed_current} | "
                f"STALE: {recomputed_stale} | UNKNOWN: {recomputed_unknown} | "
                f"Dernier HTTP: {code_str} | Collecte: {fetch_str}"
            )
            if last_err:
                lines.append(f"    [!] Dernière anomalie : {last_err}")

            if p_inds:
                for ind in p_inds:
                    name = ind.get("indicator", "UNKNOWN")
                    val = ind.get("value")
                    unit = ind.get("unit", "")
                    period = ind.get("reference_period") or "N/A"
                    eff_date = ind.get("effective_date")
                    pub_date = ind.get("published_at")
                    retrieved_date = ind.get("retrieved_at", "N/A")
                    status_db = ind.get("status", "UNKNOWN")
                    
                    stat_recalc, age_days = p_obj.evaluate_staleness(
                        indicator=name,
                        reference_period=ind.get("reference_period"),
                        effective_date=eff_date,
                        published_at=pub_date,
                    )
                    icon = "[OK]   " if stat_recalc == "OFFICIAL_CURRENT" else ("[STALE]" if stat_recalc == "STALE" else "[ALERT]")
                    val_str = f"{val} {unit}"
                    age_str = f"{age_days:.0f}j" if age_days < 999 else "N/A"
                    ref_disp = str(period) if period != "N/A" else (str(eff_date)[:10] if eff_date else "N/A")
                    coll_disp = str(retrieved_date)[:10] if retrieved_date != "N/A" else "N/A"
                    lines.append(
                        f"    {icon} {name:<23} : {val_str:<16} | Ref: {ref_disp:<10} | Coll: {coll_disp:<10} | "
                        f"Age: {age_str:<5} [BD: {status_db} | Recalc: {stat_recalc}]"
                    )
            else:
                lines.append("    (Aucun indicateur disponible en mémoire locale)")
            lines.append("")

        lines.append("2. CALENDRIER ECONOMIQUE & COUVERTURE TEMPORELLE")
        lines.append("-" * 115)
        cal_summary = self.calendar.get_calendar_coverage_summary() if hasattr(self.calendar, "get_calendar_coverage_summary") else {}
        blackout_res = self.calendar.is_in_news_blackout(symbol="EURUSD", window_minutes=30.0)
        
        cal_feed_type = cal_summary.get("feed_type", blackout_res.get("calendar_status", "LOCAL_FALLBACK"))
        tot_events = cal_summary.get("total_events", 0)
        verif_events = cal_summary.get("verified_count", 0)
        estim_events = cal_summary.get("estimated_count", 0)
        min_date = cal_summary.get("min_date") or "N/A"
        max_date = cal_summary.get("max_date") or "N/A"
        fut_days = cal_summary.get("future_coverage_days", 0.0)
        fut_events = cal_summary.get("future_events_count", 0)
        cov_status = cal_summary.get("coverage_status", "INSUFFICIENT")

        next_ev = cal_summary.get("next_event")
        if next_ev:
            next_title = next_ev.get("title", "Événement inconnu")
            next_curr = next_ev.get("currency", "USD")
            next_time = str(next_ev.get("event_time", ""))[:16]
            next_conf = next_ev.get("calendar_confidence", "APPROXIMATE")
            next_str = f"{next_title} ({next_curr}) le {next_time} UTC [{next_conf}]"
        else:
            next_str = "Aucun événement futur enregistré"

        in_blk = blackout_res.get("in_blackout", False)
        blk_event = blackout_res.get("event_title") or "Aucune annonce majeure imminente"
        mins = blackout_res.get("minutes_to_event", 999.0)

        lines.append(f"  * Source Flux          : {cal_feed_type} (Total: {tot_events} evts | Certifiés: {verif_events} | Estimés: {estim_events})")
        lines.append(f"  * Période Couverte     : {str(min_date)[:10]} -> {str(max_date)[:10]} (Horizon: {fut_days:.1f} jours / {fut_events} evts à venir) [{cov_status}]")
        lines.append(f"  * Prochain Événement   : {next_str}")
        lines.append(f"  * Fenêtre Blackout     : {'[BLOQUE] ACTIF (ORDRES INTERDITS) - ' + blk_event + f' (dans {mins:.1f} min)' if in_blk else '[OK] INACTIF (Trading toléré)'}")
        lines.append("")

        lines.append("3. MARCHES MONDIAUX, BANQUES CENTRALES (G8) & SENTIMENT GLOBAL")
        lines.append("-" * 115)
        news_articles = self.memory.get_recent_news(limit=50) if hasattr(self.memory, "get_recent_news") else []
        risk_reg = self.news.get_global_risk_regime(hours=24.0) if hasattr(self.news, "get_global_risk_regime") else {"regime": "NEUTRAL", "label": "N/A"}
        oil_s = self.news.get_sentiment_summary("OIL", hours=24.0) if hasattr(self.news, "get_sentiment_summary") else {"sentiment_label": "NEUTRAL", "article_count": 0}
        btc_s = self.news.get_sentiment_summary("BTC", hours=24.0) if hasattr(self.news, "get_sentiment_summary") else {"sentiment_label": "NEUTRAL", "article_count": 0}
        g8_rates = []
        for c_code in ("USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"):
            cb_i = self.macro.get_central_bank_info(c_code)
            g8_rates.append(f"{c_code}: {cb_i.get('rate', 0):.2f}%")
        lines.append(f"  * Taux Banques G8      : {' | '.join(g8_rates)}")
        lines.append(f"  * Climat Risque Monde  : [{risk_reg.get('regime', 'NEUTRAL')}] {risk_reg.get('label', '')}")
        lines.append(f"  * Petrole (WTI/Brent)  : {oil_s.get('sentiment_label', 'NEUTRAL')} ({oil_s.get('article_count', 0)} articles) -> Impact direct CAD")
        lines.append(f"  * Crypto Mondial 24/7  : {btc_s.get('sentiment_label', 'NEUTRAL')} ({btc_s.get('article_count', 0)} articles) -> Impact BTC/ETH/SOL")
        lines.append(f"  * Articles en mémoire  : {len(news_articles)} article(s) mondiaux analysé(s)")
        lines.append("=" * 115)
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
