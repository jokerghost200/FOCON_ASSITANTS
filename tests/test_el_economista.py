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


# ─── 9. TESTS EXIGÉS PAR L'AUDIT : 7 SCÉNARIOS CRITIQUES DE FIABILITÉ ET NON-TRANSMISSION MT5 ───

def test_scenario_1_calendrier_ne_repond_plus_fail_closed():
    """
    Scénario 1 : Le calendrier ne répond plus ou lève une exception.
    Résultat obligatoire : Aucun nouvel ordre automatique si le risque économique ne peut pas être évalué (WAIT).
    """
    fati = FATIBrain()
    fati.economista.analizar_contexto_economico = MagicMock(side_effect=RuntimeError("Calendar provider timeout / network unreachable"))

    tech = {"close": 1.0850, "rsi_14": 50.0, "atr_14": 0.0015, "trend": "Bullish"}
    decision = fati.observe_and_decide(
        symbol="EURUSD",
        timeframe="M15",
        tech_features=tech,
    )

    # Doit strictement bloquer l'ordre (Fail-Closed)
    assert decision["action"] == "WAIT"
    assert decision["confidence"] == 0
    assert "Fail-Closed" in decision["reason"] or "Gel d'ordre macroéconomique" in decision["reason"]


def test_scenario_2_annonce_reelle_imminente_wait_zero_mt5():
    """
    Scénario 2 : Une annonce réelle est imminente.
    Résultat obligatoire : WAIT, puis aucune transmission à MT5 (0 appel à mt5.order_send).
    """
    db_file = ":memory:"
    mem = EconomicMemory(db_path=db_file)
    now = datetime.datetime.now(datetime.timezone.utc)
    imminent_time = (now + datetime.timedelta(minutes=5)).isoformat()
    mem.save_economic_events([
        {
            "event_id": "imminent_nfp",
            "title": "US Non-Farm Payrolls",
            "country": "US",
            "currency": "USD",
            "date_utc": imminent_time,
            "impact": "HIGH",
        }
    ])

    cal = CalendarProvider(memory=mem)
    news = NewsCollector(memory=mem)
    macro = MacroAnalyzer(memory=mem)
    economista = ElEconomista(memory=mem, calendar=cal, news=news, macro=macro)

    econ_ctx = economista.analizar_contexto_economico("EURUSD")
    assert econ_ctx["high_impact_window"] is True
    assert econ_ctx["recommendation"] == "NO_TRADE_NEWS"

    estratega = ElEstratega()
    tech = {"close": 1.0850, "rsi_14": 52.0, "atr_14": 0.0015, "trend": "Bullish"}
    decision = estratega.decidir(
        symbol="EURUSD",
        timeframe="M15",
        tech_features=tech,
        economic_context=econ_ctx,
    )
    assert decision["action"] == "WAIT"

    # Vérification TradeExecutionEngine : barrière infranchissable
    from src.trading.trade_execution import TradeExecutionEngine
    mock_connector = MagicMock()
    mock_connector.get_account_info.return_value = {"balance": 100000.0, "equity": 100000.0}
    mock_connector.get_symbol_info.return_value = {
        "ask": 1.0855, "bid": 1.0850, "digits": 5, "trade_tick_value": 1.0, "trade_tick_size": 0.00001, "filling_mode": 1
    }
    trade_exec = TradeExecutionEngine(mt5_connector=mock_connector)

    # 1. Préparation de l'ordre refusée
    signal = {"symbol": "EURUSD", "decision": decision, "economic_context": econ_ctx}
    prep_res = trade_exec.prepare_order_request(signal)
    assert prep_res is None

    # 2. Même si une requête brute contourne et tente d'exécuter avec le contexte économique :
    raw_req = {
        "symbol": "EURUSD",
        "action": 0,
        "volume": 0.1,
        "price": 1.0855,
        "sl": 1.0800,
        "tp": 1.0950,
        "_economic_context": econ_ctx,
    }
    with patch("MetaTrader5.order_send") as mock_send:
        exec_res = trade_exec.execute_order(raw_req)
        assert exec_res["success"] is False
        assert "Gel macroéconomique" in exec_res["reason"]
        mock_send.assert_not_called()


def test_scenario_3_dates_calendrier_incertaines():
    """
    Scénario 3 : Les dates du calendrier local sont incertaines (hors réunions fixes vérifiées).
    Résultat obligatoire : Ne pas présenter le risque comme faible avec certitude (marquage APPROXIMATE).
    """
    db_file = ":memory:"
    mem = EconomicMemory(db_path=db_file)
    cal = CalendarProvider(memory=mem)

    # 1. Calendrier officiel vérifié
    events = cal.generate_recurring_fallback_events()
    assert len(events) > 0
    assert all(e.get("calendar_confidence") == "VERIFIED" for e in events)

    # 2. Insertion d'un événement approximatif / non vérifié
    now = datetime.datetime.now(datetime.timezone.utc)
    mem.save_economic_events([
        {
            "event_id": "approx_event_01",
            "title": "Estimated Retail Sales Forecast",
            "country": "US",
            "currency": "USD",
            "date_utc": (now + datetime.timedelta(minutes=10)).isoformat(),
            "impact": "HIGH",
            "is_verified": False,
            "calendar_confidence": "APPROXIMATE",
        }
    ])
    blackout_info = cal.is_in_news_blackout("EURUSD")
    assert blackout_info["in_blackout"] is True
    assert blackout_info["is_verified"] is False
    assert blackout_info["calendar_confidence"] == "APPROXIMATE"
    assert "APPROXIMATE" in blackout_info["reason"]


def test_scenario_4_mt5_insuffisant_force_devises_unavailable_sans_score_invente():
    """
    Scénario 4 : MT5 ne fournit pas assez de bougies (< 3 paires valides).
    Résultat obligatoire : Force des devises indisponible (status=UNAVAILABLE), sans score inventé.
    """
    mock_connector = MagicMock()
    # Retourne None pour toutes les requêtes de bougies
    mock_connector.get_rates_dataframe.return_value = None

    cs = CurrencyStrengthEngine(mt5_connector=mock_connector)
    bias = cs.get_pair_bias("EURUSD")

    assert bias["status"] == "UNAVAILABLE"
    assert bias["available"] is False
    assert bias["bias"] == "NEUTRAL"
    assert bias["base_strength"] == 50.0
    assert bias["quote_strength"] == 50.0
    assert bias["differential"] == 0.0
    assert "Données de cours MT5 insuffisantes" in bias["reason"]


def test_scenario_5_taux_directeur_ancien_marque_perime_carry_neutre():
    """
    Scénario 5 : Un taux directeur est ancien (> 90 jours).
    Résultat obligatoire : Donnée marquée périmée, neutralisation du carry trade.
    """
    db_file = ":memory:"
    mem = EconomicMemory(db_path=db_file)
    macro = MacroAnalyzer(memory=mem)

    # On enregistre un snapshot périmé (> 90 jours)
    old_date = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=120)).isoformat()
    macro.central_bank_rates["USD"]["last_verified_at"] = old_date
    with mem._get_connection() as conn:
        conn.execute("UPDATE macro_snapshots SET updated_at = ? WHERE currency = 'USD'", (old_date,))
        conn.commit()

    res = macro.calculate_interest_differential("USDJPY")
    assert res["is_stale"] is True
    assert res["carry_bias"] == "NEUTRAL"
    assert "périmé" in res["reason"]


def test_scenario_6_flux_rss_vides_ou_anciens_sentiment_neutre():
    """
    Scénario 6 : Les flux RSS sont vides ou anciens (> 48h).
    Résultat obligatoire : Sentiment neutre ou indisponible, jamais une fausse conviction.
    """
    db_file = ":memory:"
    mem = EconomicMemory(db_path=db_file)
    news = NewsCollector(memory=mem)

    # Base complètement vide
    summary_empty = news.get_sentiment_summary(currency="USD")
    assert summary_empty["sentiment_label"] == "NEUTRAL"
    assert summary_empty["sentiment_score"] == 0.0
    assert summary_empty["article_count"] == 0
    assert summary_empty["is_fresh"] is False

    # Article vieux de 72 heures
    old_published = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=72)).isoformat()
    mem.save_news_articles([{
        "title": "Fed announces massive interest rate cut",
        "link": "https://example.com/old",
        "published_utc": old_published,
        "currency": "USD",
        "sentiment_score": 0.9,
        "sentiment_label": "BULLISH",
    }])

    # On interroge avec une fenêtre de 24h : l'article ancien ne doit pas influencer
    summary_recent = news.get_sentiment_summary(currency="USD", hours=24.0)
    assert summary_recent["sentiment_label"] == "NEUTRAL"
    assert summary_recent["article_count"] == 0


def test_scenario_7_cerveau_economique_exception_fail_closed_zero_mt5():
    """
    Scénario 7 : Le cerveau économique lève une exception inattendue.
    Résultat obligatoire : Aucun contournement du blocage de sécurité, zero transmission à MT5.
    """
    fati = FATIBrain()
    # Simulation d'un crash brutal dans le cerveau économique
    fati.economista.analizar_contexto_economico = MagicMock(side_effect=Exception("Critical SQLite Lock Failure"))

    tech = {"close": 1.0850, "rsi_14": 50.0, "atr_14": 0.0015, "trend": "Bullish"}
    decision = fati.observe_and_decide(
        symbol="EURUSD",
        timeframe="M15",
        tech_features=tech,
    )

    # La décision doit être WAIT
    assert decision["action"] == "WAIT"
    assert decision["confidence"] == 0
    assert "Fail-Closed" in decision["reason"] or "Gel d'ordre macroéconomique" in decision["reason"]

    # Tentative d'exécution dans TradeExecutionEngine
    from src.trading.trade_execution import TradeExecutionEngine
    mock_connector = MagicMock()
    mock_connector.get_account_info.return_value = {"balance": 100000.0, "equity": 100000.0}
    mock_connector.get_symbol_info.return_value = {
        "ask": 1.0855, "bid": 1.0850, "digits": 5, "trade_tick_value": 1.0, "trade_tick_size": 0.00001, "filling_mode": 1
    }
    trade_exec = TradeExecutionEngine(mt5_connector=mock_connector)

    signal = {"symbol": "EURUSD", "decision": decision, "economic_context": decision.get("economic_context")}
    assert trade_exec.prepare_order_request(signal) is None

    # Et si ordre forcé manuellement
    forced_req = {
        "symbol": "EURUSD",
        "action": 0,
        "volume": 0.05,
        "price": 1.0855,
        "sl": 1.0800,
        "tp": 1.0950,
        "_economic_context": decision.get("economic_context"),
    }
    with patch("MetaTrader5.order_send") as mock_send:
        res = trade_exec.execute_order(forced_req)
        assert res["success"] is False
        mock_send.assert_not_called()
