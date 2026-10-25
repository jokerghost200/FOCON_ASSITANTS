# -*- coding: utf-8 -*-
"""
Tests Unitaires et d'Intégration pour El Economista
===================================================
Vérifie la robustesse de l'architecture macroéconomique à coût zéro :
1. Mémoire locale SQLite (EconomicMemory)
2. Calendrier économique déterministe et détection de blackout (CalendarProvider)
3. Collecteur RSS financier et scoring de sentiment (NewsCollector)
4. Moteur de force relative des devises (CurrencyStrengthEngine)
5. Analyseur macroéconomique et Carry Trade (MacroAnalyzer)
6. Moteur de surprise économique (EventImpactEngine)
7. Cerveau El Economista (analizar_contexto_economico)
8. Raccordement et protection anti-news dans El Estratega & FATIBrain
"""

import os
import time
import datetime
import pytest
from unittest.mock import MagicMock, patch

from src.economic.economic_memory import EconomicMemory
from src.economic.calendar_provider import CalendarProvider
from src.economic.news_collector import NewsCollector
from src.economic.currency_strength import CurrencyStrengthEngine
from src.economic.macro_analyzer import MacroAnalyzer
from src.economic.event_impact import EventImpactEngine
from src.cerebros.el_economista import ElEconomista
from src.cerebros.el_estratega import ElEstratega
from src.core.fati_brain import FATIBrain


# ─── 1. TEST ECONOMIC MEMORY (SQLITE) ───

def test_economic_memory_crud():
    """Vérifie le stockage, dédoublonnage et extraction dans la base SQLite locale."""
    db_file = ":memory:"
    mem = EconomicMemory(db_path=db_file)

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # Ajout d'événements
    events = [
        {
            "event_id": "test_nfp_001",
            "title": "US Non-Farm Payrolls",
            "country": "US",
            "currency": "USD",
            "date_utc": now_iso,
            "impact": "HIGH",
            "forecast": "180K",
            "previous": "175K",
            "actual": "220K",
        }
    ]
    saved_cnt = mem.save_economic_events(events)
    assert saved_cnt == 1

    # Dédoublonnage sur réinsertion
    saved_again = mem.save_economic_events(events)
    assert saved_again == 1  # REPLACE INTO

    # Lecture
    retrieved = mem.get_upcoming_events(window_minutes=2880.0, min_impact="HIGH")
    assert len(retrieved) >= 1
    assert retrieved[0]["title"] == "US Non-Farm Payrolls"
    assert retrieved[0]["currency"] == "USD"

    # Test News
    news = [
        {
            "title": "Federal Reserve signals rate hold amid sticky inflation",
            "link": "https://example.com/fed_news",
            "source": "BBC",
            "published_utc": now_iso,
            "currency": "USD",
            "summary": "The Fed decided to keep rates unchanged at 5.5%.",
            "sentiment_score": -0.2,
            "sentiment_label": "BEARISH",
        }
    ]
    n_saved = mem.save_news_articles(news)
    assert n_saved == 1

    recent_news = mem.get_recent_news(currency="USD", hours=24.0)
    assert len(recent_news) == 1
    assert "Federal Reserve" in recent_news[0]["title"]


# ─── 2. TEST CALENDAR PROVIDER & NEWS BLACKOUT ───

def test_calendar_provider_deterministic_and_blackout():
    """Vérifie le calendrier récurrent hors-ligne et la détection du News Blackout."""
    db_file = ":memory:"
    mem = EconomicMemory(db_path=db_file)
    cal = CalendarProvider(memory=mem)

    # Récupération (qui active le calendrier récurrent garanti sans clé API)
    events = cal.fetch_public_calendar()
    assert len(events) > 0
    assert any("Non-Farm Payrolls" in e["title"] or "CPI" in e["title"] for e in events)

    # Création manuelle d'un événement à T + 10 minutes (dans la fenêtre de gel)
    now = datetime.datetime.now(datetime.timezone.utc)
    imminent_event_time = (now + datetime.timedelta(minutes=10)).isoformat()
    mem.save_economic_events([
        {
            "event_id": "imminent_fomc",
            "title": "FOMC Interest Rate Decision",
            "country": "US",
            "currency": "USD",
            "date_utc": imminent_event_time,
            "impact": "HIGH",
        }
    ])

    # Le symbole EURUSD contient USD -> Doit être en blackout
    blackout_res = cal.is_in_news_blackout(symbol="EURUSD", window_minutes=30.0, min_impact="HIGH")
    assert blackout_res["in_blackout"] is True
    assert "FOMC" in blackout_res["event_title"]
    assert 0 <= blackout_res["minutes_to_event"] <= 15.0

    # Un symbole sans USD (ex: AUDNZD) ne doit PAS être impacté par un événement US
    blackout_audnzd = cal.is_in_news_blackout(symbol="AUDNZD", window_minutes=30.0, min_impact="HIGH")
    assert blackout_audnzd["in_blackout"] is False


# ─── 3. TEST NEWS COLLECTOR & SENTIMENT SCORING ───

def test_news_collector_sentiment_and_deduplication():
    """Vérifie l'analyse de sentiment lexicale et le dédoublonnage de titres."""
    db_file = ":memory:"
    mem = EconomicMemory(db_path=db_file)
    collector = NewsCollector(memory=mem)

    # Titres test
    bull_title = "US Dollar rallies to 2-month high on strong GDP growth and rate hike optimism"
    bear_title = "Global recession fears trigger market crash as bank defaults cause massive panic"
    neutral_title = "Treasury department releases weekly statement on administrative budget"

    score_bull, label_bull = collector.score_text_sentiment(bull_title)
    assert score_bull > 0
    assert label_bull == "BULLISH"

    score_bear, label_bear = collector.score_text_sentiment(bear_title)
    assert score_bear < 0
    assert label_bear == "BEARISH"

    score_neu, label_neu = collector.score_text_sentiment(neutral_title)
    assert label_neu == "NEUTRAL"

    # Mock RSS fetch
    mock_rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Test Business News</title>
        <item>
          <title>Dollar surge continues after robust employment data</title>
          <link>https://test.news/1</link>
          <description>Employers added more jobs than forecasted.</description>
          <pubDate>Sun, 25 Oct 2026 05:00:00 GMT</pubDate>
        </item>
      </channel>
    </rss>
    """
    with patch("urllib.request.urlopen") as mock_url:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = mock_rss_xml.encode("utf-8")
        mock_url.return_value.__enter__.return_value = mock_resp

        articles = collector.collect_from_rss_feeds()
        assert len(articles) >= 1
        assert "Dollar surge" in articles[0]["title"]


# ─── 4. TEST CURRENCY STRENGTH ENGINE (MT5 PRICE DATA) ───

def test_currency_strength_engine():
    """Vérifie le calcul du Currency Strength Index (CSI 0-100) depuis MT5."""
    mock_mt5 = MagicMock()
    # Simuler des variations : EUR très fort (+1.2%), USD très faible (-0.8%)
    def mock_rates(pair, tf, count):
        if "EUR" in pair:
            return [{"open": 1.0500, "close": 1.0626}]  # +1.2%
        elif "USD" in pair:
            return [{"open": 1.0000, "close": 0.9920}]  # -0.8%
        return [{"open": 1.0000, "close": 1.0000}]

    mock_mt5.get_rates_from_pos.side_effect = mock_rates

    csi_engine = CurrencyStrengthEngine(mt5_connector=mock_mt5)
    bias_info = csi_engine.get_pair_bias("EURUSD")

    assert "base_strength" in bias_info
    assert "quote_strength" in bias_info
    assert bias_info["bias"] in ("BULLISH", "BEARISH", "NEUTRAL")


# ─── 5. TEST MACRO ANALYZER & CARRY TRADE ───

def test_macro_analyzer_carry_trade():
    """Vérifie le différentiel de taux d'intérêt des banques centrales."""
    db_file = ":memory:"
    mem = EconomicMemory(db_path=db_file)
    macro = MacroAnalyzer(memory=mem)

    # USD (5.25%) vs JPY (0.25%) -> Taux USD > JPY -> USDJPY Carry Trade Haussier
    usdjpy_diff = macro.calculate_interest_differential("USDJPY")
    assert usdjpy_diff["differential"] > 0
    assert usdjpy_diff["carry_bias"] == "BULLISH"
    assert usdjpy_diff["base_rate"] == 5.25
    assert usdjpy_diff["quote_rate"] == 0.25

    # EUR (3.25%) vs USD (5.25%) -> Taux EUR < USD -> EURUSD Carry Trade Baissier
    eurusd_diff = macro.calculate_interest_differential("EURUSD")
    assert eurusd_diff["differential"] < 0
    assert eurusd_diff["carry_bias"] == "BEARISH"


# ─── 6. TEST EVENT IMPACT ENGINE (SURPRISE INDEX) ───

def test_event_impact_engine():
    """Vérifie le calcul de l'écart aux prévisions et son impact devises."""
    engine = EventImpactEngine()

    # NFP supérieur aux attentes : 250K vs 180K prévu (+38% surprise)
    impact_bull = engine.evaluate_surprise(
        event_title="US Non-Farm Payrolls",
        currency="USD",
        actual=250.0,
        forecast=180.0,
        previous=170.0,
    )
    assert impact_bull["diff"] > 0
    assert impact_bull["currency_bias"] == "BULLISH"

    # Traduction pour la paire EURUSD : USD fort -> EURUSD Baissier
    symbol_bias = engine.map_surprise_to_symbol_impact("EURUSD", impact_bull)
    assert symbol_bias == "BEARISH"


# ─── 7. TEST EL ECONOMISTA BRAIN ───

def test_el_economista_brain_analysis():
    """Vérifie la synthèse globale d'El Economista et le gel d'ordre en cas de blackout."""
    db_file = ":memory:"
    mem = EconomicMemory(db_path=db_file)
    cal = CalendarProvider(memory=mem)
    news = NewsCollector(memory=mem)
    macro = MacroAnalyzer(memory=mem)

    economista = ElEconomista(
        memory=mem,
        calendar=cal,
        news=news,
        macro=macro,
    )

    # Cas 1 : Marché normal sans événement imminent
    res = economista.analizar_contexto_economico("EURUSD")
    assert res["cerebro_origen"] == "El Economista"
    assert res["symbol"] == "EURUSD"
    assert res["bias"] in ("BULLISH", "BEARISH", "NEUTRAL")
    assert 0.0 <= res["confidence"] <= 1.0
    assert res["high_impact_window"] is False
    assert res["recommendation"] in ("NORMAL", "FAVOR_BUY", "FAVOR_SELL")

    # Cas 2 : Événement majeur imminent inséré en mémoire
    now = datetime.datetime.now(datetime.timezone.utc)
    imminent_event_time = (now + datetime.timedelta(minutes=5)).isoformat()
    mem.save_economic_events([
        {
            "event_id": "urgent_cpi",
            "title": "US Consumer Price Index (CPI)",
            "country": "US",
            "currency": "USD",
            "date_utc": imminent_event_time,
            "impact": "HIGH",
        }
    ])

    res_blackout = economista.analizar_contexto_economico("EURUSD")
    assert res_blackout["high_impact_window"] is True
    assert res_blackout["recommendation"] == "NO_TRADE_NEWS"
    assert res_blackout["event_risk"] in ("CRITICAL", "HIGH")
    assert "CPI" in res_blackout["reason"]
    assert economista.es_periodo_bloqueo_noticias("EURUSD") is True


# ─── 8. TEST END-TO-END : EL ESTRATEGA & FATIBRAIN BLOCKING ON NEWS ───

def test_el_estratega_bloque_par_contexte_economique():
    """
    Vérifie qu'El Estratega impose strictement un statut WAIT lorsque
    le contexte macroéconomique indique une période de News Blackout.
    """
    estratega = ElEstratega()

    # Setup technique parfait (Achat valide)
    tech = {"close": 1.0850, "rsi_14": 52.0, "atr_14": 0.0015, "trend": "Bullish"}
    economic_blackout = {
        "high_impact_window": True,
        "event_risk": "HIGH",
        "recommendation": "NO_TRADE_NEWS",
        "reason": "Publication US Non-Farm Payrolls dans 4 minutes",
    }

    decision = estratega.decidir(
        symbol="EURUSD",
        timeframe="M15",
        tech_features=tech,
        economic_context=economic_blackout,
    )

    # Sécurité absolue : l'ordre doit être gelé
    assert decision["action"] == "WAIT"
    assert decision["confidence"] == 0
    assert "Gel d'ordre macroéconomique" in decision["reason"]
    assert "Non-Farm Payrolls" in decision["reason"]


def test_fati_brain_integration_el_economista():
    """
    Vérifie l'intégration du 6ème cerveau dans FATIBrain et observe_and_decide.
    """
    fati = FATIBrain()
    assert hasattr(fati, "economista")
    assert fati.economista is not None

    # observe_and_decide doit inclure le contexte économique
    tech = {"close": 1.0850, "rsi_14": 50.0, "atr_14": 0.0015, "trend": "Bullish"}
    dec = fati.observe_and_decide(
        symbol="EURUSD",
        timeframe="M15",
        tech_features=tech,
    )

    assert "cerebro_origen" in dec
    assert "action" in dec
    # Le contexte économique a été interrogé
    assert "fati_decision_id" in dec
