# -*- coding: utf-8 -*-
"""
FATI News Collector & Local Sentiment Analyzer (Collecteur d'Actualités RSS Gratuit)
===================================================================================
Collecte les dépêches financières en temps réel depuis des flux RSS publics ouverts :
- BBC News Business (https://feeds.bbci.co.uk/news/business/rss.xml)
- Al Jazeera (https://www.aljazeera.com/xml/rss/all.xml)
- Yahoo Finance (https://finance.yahoo.com/news/rssindex)
Analyse locale du sentiment sans clé d'API grâce à un lexique financier quantitatif.
Dédoublonnage automatique et stockage dans EconomicMemory.
"""
from __future__ import annotations

import re
import logging
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from src.economic.economic_memory import EconomicMemory

logger = logging.getLogger(__name__)

# Flux RSS publics gratuits (Couverture Mondiale : Amériques, Europe, Asie-Pacifique, Matières Premières & Crypto 24/7)
PUBLIC_RSS_FEEDS = [
    {
        "name": "BBC Business (Monde/Europe)",
        "url": "https://feeds.bbci.co.uk/news/business/rss.xml",
    },
    {
        "name": "Yahoo Finance (Wall Street/Global)",
        "url": "https://finance.yahoo.com/news/rssindex",
    },
    {
        "name": "CNBC World Economy & Markets",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    },
    {
        "name": "Investing Global Forex & Commodities",
        "url": "https://www.investing.com/rss/news_1.rss",
    },
    {
        "name": "Cointelegraph (Crypto 24/7)",
        "url": "https://cointelegraph.com/rss",
    },
    {
        "name": "Al Jazeera (Géopolitique & Énergie)",
        "url": "https://www.aljazeera.com/xml/rss/all.xml",
    },
]

# Lexique financier de sentiment (Mots-clés haussiers vs baissiers)
BULLISH_KEYWORDS = {
    "surge", "surges", "soar", "soars", "gain", "gains", "rally", "rallies", "growth",
    "jump", "jumps", "record high", "optimism", "rate hike", "hawkish", "outperform",
    "rebound", "expansion", "profit rise", "stronger", "positive", "beat forecast",
    "hiring boom", "bullish", "upturn", "boom", "stimulus", "inflow", "inflows",
    "adoption", "breakout", "recovery", "strong demand", "upbeat",
}

BEARISH_KEYWORDS = {
    "plunge", "plunges", "drop", "drops", "slump", "slumps", "fall", "falls",
    "decline", "declines", "tumble", "tumbles", "recession", "recessions", "pessimism",
    "rate cut", "dovish", "underperform", "inflation fear", "slowdown", "losses",
    "weaker", "negative", "miss forecast", "layoffs", "bearish", "crisis", "default",
    "defaults", "crash", "crashes", "collapse", "collapses", "panic", "fear", "fears", "downturn",
    "outflow", "outflows", "hack", "ban", "sanctions", "deficit", "selloff", "sell-off",
}

RISK_OFF_KEYWORDS = {
    "war", "missile", "strike", "conflict", "escalation", "geopolitical", "sanctions",
    "crisis", "panic", "crash", "collapse", "recession", "safe-haven", "safe haven",
    "tariff", "tariffs", "trade war", "default", "bank run", "emergency", "invasion",
}

RISK_ON_KEYWORDS = {
    "rally", "optimism", "record high", "soft landing", "stimulus", "peace",
    "ceasefire", "trade deal", "boom", "expansion", "bull market", "risk-on", "upbeat",
}

CURRENCY_KEYWORDS = {
    "USD": ["fed", "federal reserve", "powell", "fomc", "dollar", "us economy", "treasury", "wall street", "nfp", "u.s."],
    "EUR": ["ecb", "lagarde", "euro", "eurozone", "germany", "bundesbank", "inflation eu", "france", "frankfurt"],
    "GBP": ["boe", "bank of england", "bailey", "sterling", "pound", "uk economy", "britain", "london"],
    "JPY": ["boj", "bank of japan", "ueda", "yen", "tokyo", "nikkei", "japan"],
    "CAD": ["boc", "bank of canada", "macklem", "loonie", "canada", "canadian dollar", "ottawa"],
    "AUD": ["rba", "reserve bank of australia", "bullock", "aussie", "australia", "iron ore", "sydney", "china stimulus", "pboc"],
    "NZD": ["rbnz", "reserve bank of new zealand", "kiwi", "new zealand", "auckland", "wellington"],
    "CHF": ["snb", "swiss national bank", "swiss franc", "switzerland", "zurich"],
    "XAU": ["gold", "bullion", "safe-haven", "precious metals", "xau", "silver"],
    "OIL": ["oil", "crude", "wti", "brent", "opec", "energy", "petroleum", "barrel"],
    "BTC": ["bitcoin", "btc", "crypto", "cryptocurrency", "etf", "ethereum", "eth", "solana", "blockchain", "whale", "sec", "binance", "coinbase"],
}


class NewsCollector:
    """
    Collecteur d'actualités et analyseur de sentiment financier local autonome.
    """

    def __init__(self, memory: Optional[EconomicMemory] = None, timeout_seconds: float = 5.0):
        self.memory = memory or EconomicMemory()
        self.timeout_seconds = timeout_seconds

    def collect_from_rss_feeds(self) -> List[Dict[str, Any]]:
        """Interroge tous les flux RSS publics configurés."""
        all_articles: List[Dict[str, Any]] = []

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FoconAssistant/1.0 RSS Reader",
            "Accept": "application/rss+xml, application/xml, text/xml",
        }

        for feed in PUBLIC_RSS_FEEDS:
            url = feed["url"]
            source_name = feed["name"]
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    if resp.status == 200:
                        xml_content = resp.read()
                        articles = self._parse_rss_xml(xml_content, source_name)
                        for a in articles:
                            if self.memory.store_news(a):
                                all_articles.append(a)
                        logger.info(f"📰 [NewsCollector] {len(articles)} articles collectés depuis {source_name}.")
            except Exception as e:
                logger.debug(f"Flux RSS {source_name} indisponible ({e}).")

        return all_articles

    def _parse_rss_xml(self, xml_bytes: bytes, source: str) -> List[Dict[str, Any]]:
        """Parse le flux XML RSS et extrait les éléments pertinents."""
        articles = []
        try:
            root = ET.fromstring(xml_bytes)
            ch = root.find("channel")
            channel = ch if ch is not None else root

            for item in channel.findall("item"):
                title = item.findtext("title") or ""
                link = item.findtext("link") or ""
                description = item.findtext("description") or ""
                pub_date = item.findtext("pubDate") or datetime.now(timezone.utc).isoformat()

                if not title:
                    continue

                full_text = f"{title} {description}"
                currency = self._detect_currency(full_text)
                score, label = self.analyze_sentiment(full_text)

                articles.append({
                    "title": title.strip(),
                    "link": link.strip(),
                    "summary": description[:300].strip(),
                    "source": source,
                    "currency": currency,
                    "sentiment_score": score,
                    "sentiment_label": label,
                    "published_at": pub_date,
                })
        except Exception as e:
            logger.debug(f"Erreur parsing RSS: {e}")

        return articles

    def _detect_currency(self, text: str) -> str:
        """Détecte la devise la plus concernée par le texte."""
        text_lower = text.lower()
        scores = {}
        for curr, kw_list in CURRENCY_KEYWORDS.items():
            cnt = sum(1 for kw in kw_list if kw in text_lower)
            if cnt > 0:
                scores[curr] = cnt

        if not scores:
            return "GENERAL"
        return max(scores, key=scores.get)

    def analyze_sentiment(self, text: str) -> tuple[float, str]:
        """
        Calcule un score de sentiment quantitatif entre -1.0 et +1.0
        en comptant les occurrences contextuelles de mots-clés financiers.
        """
        text_lower = text.lower()
        bull_count = sum(1 for w in BULLISH_KEYWORDS if re.search(r'\b' + re.escape(w) + r'\b', text_lower))
        bear_count = sum(1 for w in BEARISH_KEYWORDS if re.search(r'\b' + re.escape(w) + r'\b', text_lower))

        total = bull_count + bear_count
        if total == 0:
            return 0.0, "NEUTRAL"

        score = round((bull_count - bear_count) / total, 2)
        if score >= 0.20:
            label = "BULLISH"
        elif score <= -0.20:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        return score, label

    # Alias pratique
    score_text_sentiment = analyze_sentiment


    def get_sentiment_summary(self, currency: str, hours: float = 24.0) -> Dict[str, Any]:
        """
        Génère une synthèse du sentiment de marché sur les dernières heures pour une devise.
        """
        news = self.memory.get_recent_news(currency=currency, hours=hours, limit=30)
        if not news:
            return {
                "currency": currency,
                "sentiment_score": 0.0,
                "sentiment_label": "NEUTRAL",
                "article_count": 0,
                "headline_sample": [],
                "is_fresh": False,
            }

        scores = [float(n.get("sentiment_score", 0.0) or 0.0) for n in news]
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0

        if avg_score >= 0.15:
            label = "BULLISH"
        elif avg_score <= -0.15:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        headlines = [n.get("title") for n in news[:3]]

        return {
            "currency": currency,
            "sentiment_score": avg_score,
            "sentiment_label": label,
            "article_count": len(news),
            "headline_sample": headlines,
            "is_fresh": len(news) > 0,
        }

    def get_global_risk_regime(self, hours: float = 24.0) -> Dict[str, Any]:
        """
        Analyse l'ensemble des dépêches mondiales récentes pour déterminer le Climat de Risque Mondial :
        - RISK_OFF : Tensions géopolitiques, guerres, krachs, crises -> favorise Valeurs Refuges (XAU, JPY, CHF, USD)
        - RISK_ON  : Optimisme boursier mondial, stimulus, accords -> favorise Croissance & Crypto (AUD, NZD, CAD, EUR, BTC, ETH)
        - NEUTRAL  : Climat mondial équilibré
        """
        all_news = self.memory.get_recent_news(currency=None, hours=hours, limit=60)
        if not all_news:
            return {
                "regime": "NEUTRAL",
                "score": 0.0,
                "risk_on_hits": 0,
                "risk_off_hits": 0,
                "article_count": 0,
                "label": "Climat mondial neutre (aucune depeche)",
            }

        risk_on_hits = 0
        risk_off_hits = 0
        for item in all_news:
            txt = f"{item.get('title', '')} {item.get('summary', '')}".lower()
            risk_on_hits += sum(1 for w in RISK_ON_KEYWORDS if w in txt)
            risk_off_hits += sum(1 for w in RISK_OFF_KEYWORDS if w in txt)

        total_hits = risk_on_hits + risk_off_hits
        if total_hits == 0:
            avg_sent = sum(float(n.get("sentiment_score", 0.0) or 0.0) for n in all_news) / len(all_news)
            score = round(avg_sent, 2)
        else:
            score = round((risk_on_hits - risk_off_hits) / total_hits, 2)

        if score <= -0.20 or (risk_off_hits >= 4 and risk_off_hits >= risk_on_hits * 1.5):
            regime = "RISK_OFF"
            label = f"RISK-OFF Mondial (Refuges XAU/JPY/CHF/USD favorises | {risk_off_hits} alertes risque)"
        elif score >= 0.20 or (risk_on_hits >= 4 and risk_on_hits >= risk_off_hits * 1.5):
            regime = "RISK_ON"
            label = f"RISK-ON Mondial (Appetit pour le risque AUD/NZD/CAD/Crypto | {risk_on_hits} signaux positifs)"
        else:
            regime = "NEUTRAL"
            label = f"Climat de risque mondial equilibre ({len(all_news)} articles)"

        return {
            "regime": regime,
            "score": score,
            "risk_on_hits": risk_on_hits,
            "risk_off_hits": risk_off_hits,
            "article_count": len(all_news),
            "label": label,
        }
