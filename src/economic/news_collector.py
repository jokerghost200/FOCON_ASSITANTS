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

# Flux RSS publics gratuits
PUBLIC_RSS_FEEDS = [
    {
        "name": "BBC Business",
        "url": "https://feeds.bbci.co.uk/news/business/rss.xml",
    },
    {
        "name": "Al Jazeera",
        "url": "https://www.aljazeera.com/xml/rss/all.xml",
    },
    {
        "name": "Yahoo Finance",
        "url": "https://finance.yahoo.com/news/rssindex",
    },
]

# Lexique financier de sentiment (Mots-clés haussiers vs baissiers)
BULLISH_KEYWORDS = {
    "surge", "surges", "soar", "soars", "gain", "gains", "rally", "rallies", "growth",
    "jump", "jumps", "record high", "optimism", "rate hike", "hawkish", "outperform",
    "rebound", "expansion", "profit rise", "stronger", "positive", "beat forecast",
    "hiring boom", "bullish", "upturn", "boom",
}

BEARISH_KEYWORDS = {
    "plunge", "plunges", "drop", "drops", "slump", "slumps", "fall", "falls",
    "decline", "declines", "tumble", "tumbles", "recession", "recessions", "pessimism",
    "rate cut", "dovish", "underperform", "inflation fear", "slowdown", "losses",
    "weaker", "negative", "miss forecast", "layoffs", "bearish", "crisis", "default",
    "defaults", "crash", "crashes", "collapse", "collapses", "panic", "fear", "fears", "downturn",
}

CURRENCY_KEYWORDS = {
    "USD": ["fed", "federal reserve", "powell", "dollar", "us economy", "treasury", "wall street", "nfp"],
    "EUR": ["ecb", "lagarde", "euro", "eurozone", "germany", "bundesbank", "inflation eu"],
    "GBP": ["boe", "bank of england", "bailey", "sterling", "pound", "uk economy"],
    "JPY": ["boj", "bank of japan", "ueda", "yen", "tokyo", "nikkei"],
    "XAU": ["gold", "bullion", "safe-haven", "precious metals", "xau"],
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
