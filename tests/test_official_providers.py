# -*- coding: utf-8 -*-
"""
Tests Unitaires & de Résilience pour les Connecteurs Économiques Officiels
========================================================================
Valide le fonctionnement des connecteurs gratuits (BLS, BCE, Eurostat)
et leur intégration dans EconomicMemory, MacroAnalyzer et ElEconomista :
1. Réponse valide (parsing et typage)
2. Source HTTP indisponible (500, 503, URLError)
3. Dépassement de délai (Timeout)
4. JSON malformé / corrompu
5. Données vides ou valeurs non numériques (ex: '-' lors d'arrêts administratifs)
6. Détection de données périmées (STALE)
7. Valeurs révisées et dédoublonnage SQLite
8. Collecte en arrière-plan sans bloquer
9. Maintien strict du Fail-Closed et du News Blackout
"""
from __future__ import annotations

import json
import sqlite3
import urllib.error
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

from src.economic.economic_memory import EconomicMemory
from src.economic.providers.base_provider import BaseEconomicProvider
from src.economic.providers.bls_provider import BLSProvider
from src.economic.providers.ecb_provider import ECBProvider
from src.economic.providers.eurostat_provider import EurostatProvider
from src.economic.macro_analyzer import MacroAnalyzer
from src.cerebros.el_economista import ElEconomista
from src.trading.trade_execution import TradeExecutionEngine


# ── Fixtures ──

@pytest.fixture
def mem_db():
    """Mémoire SQLite isolée en mémoire vive."""
    return EconomicMemory(db_path=":memory:")


# ── Tests BaseEconomicProvider ──

def test_base_provider_retry_and_timeout():
    """Vérifie la gestion des erreurs réseau et des retries sans crash."""
    provider = BaseEconomicProvider(name="TEST_PROV", timeout_seconds=0.1, max_retries=1, backoff_factor=0.01)

    # Simulation d'erreur réseau
    with patch("urllib.request.urlopen", side_effect=Exception("Connexion impossible")):
        res = provider._http_request("https://fake.service/api")
        assert res is None
        assert provider.last_status == "ERROR"

    # Simulation de payload JSON invalide
    assert provider._safe_json_loads(b"NOT_A_JSON") is None
    assert provider._safe_json_loads(None) is None

    # Payload valide
    valid = provider._safe_json_loads(b'{"status": "OK"}')
    assert valid == {"status": "OK"}


# ── Tests BLSProvider ──

def test_bls_provider_parsing_and_yoy(mem_db):
    """Vérifie le parsing BLS v1 et le calcul automatique du CPI YoY."""
    provider = BLSProvider(memory=mem_db, timeout_seconds=1.0)

    # Mock de données BLS avec Chômage, Payrolls et 13 mois de CPI
    cpi_data = [{"year": "2025", "period": "M12", "periodName": "December", "value": "320.0"}]
    for m in range(11, 0, -1):
        cpi_data.append({"year": "2025", "period": f"M{m:02d}", "periodName": f"Month{m}", "value": "315.0"})
    cpi_data.append({"year": "2024", "period": "M12", "periodName": "December", "value": "310.0"})

    mock_bls_resp = {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [
                {
                    "seriesID": "LNS14000000",
                    "data": [
                        {"year": "2025", "period": "M12", "periodName": "December", "value": "4.3"},
                        {"year": "2025", "period": "M11", "periodName": "November", "value": "4.4"},
                    ],
                },
                {
                    "seriesID": "CUUR0000SA0",
                    "data": cpi_data,
                },
                {
                    "seriesID": "CES0000000001",
                    "data": [
                        {"year": "2025", "period": "M12", "periodName": "December", "value": "158000"},
                    ],
                },
            ]
        },
    }

    with patch.object(provider, "_http_request", return_value=json.dumps(mock_bls_resp).encode("utf-8")):
        inds = provider.fetch_latest_indicators(2024, 2025)
        assert len(inds) == 4  # Unemployment, CPI_Index, Payrolls + CPI_YoY

        unemp = mem_db.get_latest_official_indicator("UNEMPLOYMENT", "USD")
        assert unemp is not None
        assert unemp["value"] == 4.3
        assert unemp["unit"] == "PERCENT"
        assert unemp["source"] == "BLS"

        cpi_yoy = mem_db.get_latest_official_indicator("CPI_YOY", "USD")
        assert cpi_yoy is not None
        # (320 - 310) / 310 * 100 = 3.23%
        assert cpi_yoy["value"] == 3.23
        assert cpi_yoy["unit"] == "PERCENT"


def test_bls_provider_handles_missing_values(mem_db):
    """Vérifie le filtrage des valeurs '-' (arrêts gouvernementaux) sans planter."""
    provider = BLSProvider(memory=mem_db, timeout_seconds=1.0)

    mock_resp = {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [
                {
                    "seriesID": "LNS14000000",
                    "data": [
                        {"year": "2025", "period": "M10", "value": "-"},
                        {"year": "2025", "period": "M09", "value": "4.1"},
                    ],
                }
            ]
        },
    }

    with patch.object(provider, "_http_request", return_value=json.dumps(mock_resp).encode("utf-8")):
        inds = provider.fetch_latest_indicators(2024, 2025)
        assert len(inds) == 1
        assert inds[0]["value"] == 4.1
        assert inds[0]["reference_period"] == "2025M09"


# ── Tests ECBProvider ──

def test_ecb_provider_sdmx_parsing(mem_db):
    """Vérifie le parsing standard SDMX 1.0 JSON de la BCE."""
    provider = ECBProvider(memory=mem_db, timeout_seconds=1.0)

    mock_sdmx_json = {
        "structure": {
            "dimensions": {
                "observation": [
                    {"values": [{"id": "2025-09-15"}, {"id": "2025-10-23"}]}
                ]
            }
        },
        "dataSets": [
            {
                "series": {
                    "0:0:0:0": {
                        "observations": {
                            "0": [3.25, 0, 0],
                            "1": [3.00, 0, 0],
                        }
                    }
                }
            }
        ],
    }

    with patch.object(provider, "_http_request", return_value=json.dumps(mock_sdmx_json).encode("utf-8")):
        inds = provider.fetch_latest_indicators(last_n_observations=2)
        assert len(inds) == 4  # 4 séries configurées

        dfr = mem_db.get_latest_official_indicator("DEPOSIT_FACILITY_RATE", "EUR")
        assert dfr is not None
        assert dfr["value"] == 3.00
        assert dfr["reference_period"] == "2025-10-23"
        assert dfr["source"] == "ECB"


# ── Tests EurostatProvider ──

def test_eurostat_provider_multi_geo_fallback(mem_db):
    """Vérifie la résolution dynamique des codes géographiques (EA21 -> EA20 -> EA)."""
    provider = EurostatProvider(memory=mem_db, timeout_seconds=1.0)

    mock_eurostat_json = {
        "value": {"0": 6.3, "1": 6.2},
        "dimension": {
            "time": {
                "category": {
                    "label": {"0": "2025-11", "1": "2025-12"}
                }
            }
        },
    }

    with patch.object(provider, "_http_request", return_value=json.dumps(mock_eurostat_json).encode("utf-8")):
        inds = provider.fetch_latest_indicators()
        assert len(inds) == 3

        une = mem_db.get_latest_official_indicator("UNEMPLOYMENT", "EUR")
        assert une is not None
        assert une["value"] == 6.2
        assert une["reference_period"] == "2025-12"
        assert une["source"] == "EUROSTAT"


# ── Tests EconomicMemory CRUD & Dédoublonnage ──

def test_economic_memory_upsert_and_filters(mem_db):
    """Vérifie le stockage, le dédoublonnage et le filtrage temporel des indicateurs."""
    ind1 = {
        "source": "BLS",
        "indicator": "UNEMPLOYMENT",
        "currency_or_region": "USD",
        "value": 4.2,
        "unit": "PERCENT",
        "reference_period": "2025M11",
        "published_at": "2025-12-05T13:30:00+00:00",
        "status": "OFFICIAL_HISTORICAL",
    }
    assert mem_db.store_official_indicator(ind1) is True

    # Révision de la même période (même id = BLS_UNEMPLOYMENT_USD_2025M11)
    ind1_revised = dict(ind1)
    ind1_revised["value"] = 4.1
    ind1_revised["revision"] = "REVISED"
    assert mem_db.store_official_indicator(ind1_revised) is True

    latest = mem_db.get_latest_official_indicator("UNEMPLOYMENT", "USD")
    assert latest["value"] == 4.1
    assert latest["revision"] == "REVISED"

    # Vérification que le dédoublonnage a bien évité la création d'une seconde ligne
    all_inds = mem_db.get_official_indicators("USD")
    assert len(all_inds) == 1


# ── Tests MacroAnalyzer & Intégration Données Réelles ──

def test_macro_analyzer_updates_with_official_data(mem_db):
    """Vérifie que MacroAnalyzer intègre automatiquement les indicateurs réels s'ils existent."""
    # Enregistrement d'un taux BCE officiel récent
    now_str = datetime.now(timezone.utc).isoformat()
    mem_db.store_official_indicator({
        "source": "ECB",
        "indicator": "DEPOSIT_FACILITY_RATE",
        "currency_or_region": "EUR",
        "value": 2.75,
        "reference_period": "2026-01-15",
        "published_at": now_str,
        "retrieved_at": now_str,
    })

    analyzer = MacroAnalyzer(memory=mem_db, max_age_days=90)
    info_eur = analyzer.get_central_bank_info("EUR")

    assert info_eur["rate"] == 2.75
    assert "ECB" in info_eur["source"]
    assert info_eur["is_stale"] is False
    assert info_eur["freshness"] == "FRESH"


# ── Tests ElEconomista & Maintien Fail-Closed ──

def test_el_economista_actualizar_indicadores(mem_db):
    """Vérifie la méthode de synchronisation globale des indicateurs officiels."""
    eco = ElEconomista(memory=mem_db)

    with patch.object(eco.bls, "fetch_latest_indicators", return_value=[{"id": "1"}, {"id": "2"}]), \
         patch.object(eco.ecb, "fetch_latest_indicators", return_value=[{"id": "3"}]), \
         patch.object(eco.eurostat, "fetch_latest_indicators", return_value=[{"id": "4"}]):

        res = eco.actualizar_indicadores_oficiales()
        assert res["status"] == "SUCCESS"
        assert res["total_indicators"] == 4
        assert res["bls_count"] == 2
        assert res["ecb_count"] == 1
        assert res["eurostat_count"] == 1


def test_el_economista_news_blackout_cannot_be_bypassed(mem_db):
    """
    Règle absolue : même si des données officielles fraîches sont présentes,
    le News Blackout DOIT bloquer les ordres (recommandation = NO_TRADE_NEWS).
    """
    eco = ElEconomista(memory=mem_db)

    # Simulation d'une annonce imminente dans 5 minutes
    with patch.object(eco.calendar, "is_in_news_blackout", return_value={
        "in_blackout": True,
        "impact": "CRITICAL",
        "reason": "Annonce imminente de taux FOMC dans 5.0 min",
        "calendar_confidence": "VERIFIED",
        "minutes_to_event": 5.0,
        "event_title": "FOMC Rate Decision",
    }):
        context = eco.analizar_contexto_economico("EURUSD")

        assert context["high_impact_window"] is True
        assert context["event_risk"] == "CRITICAL"
        assert context["recommendation"] == "NO_TRADE_NEWS"
        assert "GEL ANNONCE ÉCONOMIQUE" in context["reason"]
        assert "official_indicators" in context


def test_staleness_and_unknown_edge_cases():
    """
    Vérifie les cas critiques de evaluate_staleness et normalize_indicator :
    - Une période inconnue/corrompue ne doit JAMAIS retourner OFFICIAL_CURRENT (retourne UNKNOWN).
    - Une date future invalide ne doit pas produire un âge nul accepté (retourne UNKNOWN).
    - Les périodes mensuelles et trimestrielles calculent un âge exact par rapport à la fin de période.
    - Une valeur non numérique ou NaN/inf est classée UNKNOWN.
    """
    prov = BaseEconomicProvider(name="TEST")

    # 1. Périodes inconnues ou malformées -> UNKNOWN
    status_bad, age_bad = prov.evaluate_staleness("CPI", "INVALID_FORMAT")
    assert status_bad == "UNKNOWN"
    assert age_bad == 999.0

    status_empty, age_empty = prov.evaluate_staleness("CPI", "")
    assert status_empty == "UNKNOWN"
    assert age_empty == 999.0

    # 2. Dates futures invalides -> UNKNOWN (sauf taux directeurs imminents <= 7 jours)
    status_future, age_future = prov.evaluate_staleness("CPI", "2035M01")
    assert status_future == "UNKNOWN"
    assert age_future == 999.0

    status_future_rate, age_future_rate = prov.evaluate_staleness("DEPOSIT_FACILITY_RATE", "2035-01-01", effective_date="2035-01-01")
    assert status_future_rate == "UNKNOWN"
    assert age_future_rate == 999.0

    # 3. Données anciennes (ex: 2023) -> STALE avec âge élevé
    status_old, age_old = prov.evaluate_staleness("CPI", "2023M01")
    assert status_old == "STALE"
    assert age_old > 365.0

    # 4. Association et calcul précis des âges mensuels et trimestriels
    from datetime import date
    today = datetime.now(timezone.utc).date()

    # Mensuel BLS 2026M09 -> fin de mois 2026-09-30
    st_m9, age_m9 = prov.evaluate_staleness("CPI", "2026M09")
    expected_m9_age = float((today - date(2026, 9, 30)).days)
    assert age_m9 == expected_m9_age

    # Trimestriel Eurostat 2026-Q2 -> fin de trimestre 2026-06-30
    st_q2, age_q2 = prov.evaluate_staleness("GDP_GROWTH_YOY", "2026-Q2")
    expected_q2_age = float((today - date(2026, 6, 30)).days)
    assert age_q2 == expected_q2_age
    assert age_q2 > 0.0  # Ne doit plus jamais valoir 0.0 artificiellement !

    # 5. Valeurs non numériques ou NaN dans normalize_indicator -> UNKNOWN
    ind_nan = prov.normalize_indicator(
        indicator="CPI",
        currency_or_region="USD",
        value=float("nan"),
        unit="PERCENT",
        reference_period="2026M09",
        source_url="http://fake",
    )
    assert ind_nan["status"] == "UNKNOWN"
    assert ind_nan["age_days"] == 999.0


def test_trade_execution_engine_blocks_unknown_and_stale():
    """
    Vérifie que TradeExecutionEngine en mode automatique/semi-automatique bloque
    rigoureusement les ordres si les données économiques critiques sont UNKNOWN ou STALE,
    aussi bien au niveau de prepare_order_request() que de execute_order() avant mt5.order_send().
    """
    mock_mt5 = MagicMock()
    mock_mt5.get_open_positions.return_value = []
    mock_mt5.get_daily_profit.return_value = 0.0
    mock_mt5.account_info.return_value = MagicMock(balance=10000.0, equity=10000.0)

    mock_centinela = MagicMock()
    mock_centinela.evaluar_seguridad.return_value = {"safe": True, "reason": "OK"}
    mock_centinela.parada_emergencia_activa = False

    engine = TradeExecutionEngine(mode="automatic", mt5_connector=mock_mt5, centinela=mock_centinela)

    base_signal = {
        "symbol": "EURUSD",
        "decision": {
            "action": "BUY",
            "confidence": 0.85,
            "lot": 0.1,
            "stop_loss_pips": 20.0,
            "take_profit_pips": 40.0,
        },
    }

    # Cas 1 : Contexte économique avec status = UNKNOWN -> Rejet
    sig_unknown = dict(base_signal)
    sig_unknown["economic_context"] = {
        "status": "UNKNOWN",
        "data_freshness": "UNKNOWN",
        "reason": "Donnée macroéconomique critique impossible à vérifier",
    }
    order_req = engine.prepare_order_request(sig_unknown)
    assert order_req is None, "L'ordre aurait dû être rejeté pour contexte UNKNOWN"

    # Cas 2 : Contexte économique avec data_freshness = STALE et is_rate_stale = True -> Rejet
    sig_stale = dict(base_signal)
    sig_stale["economic_context"] = {
        "status": "OK",
        "data_freshness": "STALE",
        "is_rate_stale": True,
        "reason": "Taux directeur non vérifié depuis > 90 jours",
    }
    order_req_stale = engine.prepare_order_request(sig_stale)
    assert order_req_stale is None, "L'ordre aurait dû être rejeté pour données périmées (STALE)"

    # Cas 3 : Tentative d'appel direct à execute_order avec métadonnées UNKNOWN -> Rejet avant mt5.order_send
    direct_req = {
        "action": 1,
        "symbol": "EURUSD",
        "volume": 0.1,
        "type": 0,
        "price": 1.1000,
        "sl": 1.0980,
        "tp": 1.1040,
        "_economic_context": {
            "status": "UNKNOWN",
            "data_freshness": "UNKNOWN",
            "reason": "Taux directeurs suspects ou invalides",
        },
    }
    res = engine.execute_order(direct_req)
    assert res is not None
    assert res["success"] is False
    assert "Sécurité macroéconomique non satisfaite" in res["reason"] or "UNKNOWN" in res["reason"]
    # Vérification absolue que mt5.order_send n'a JAMAIS été appelé
    mock_mt5.order_send.assert_not_called()


def test_failed_request_preserves_stale_status_without_rejuvenation(mem_db):
    """
    Vérifie qu'une panne réseau ou une réponse vide ne remplace jamais une ancienne date
    par la date courante (pas de rajeunissement artificiel en cas d'erreur).
    """
    # Donnée historique enregistrée en 2024
    mem_db.store_official_indicator({
        "source": "BLS",
        "indicator": "UNEMPLOYMENT",
        "currency_or_region": "USD",
        "value": 3.7,
        "reference_period": "2024M01",
        "published_at": "2024-02-02T13:30:00+00:00",
        "retrieved_at": "2024-02-02T14:00:00+00:00",
        "status": "STALE",
    })

    bls = BLSProvider(memory=mem_db, timeout_seconds=0.1)

    # Simulation d'une panne réseau HTTP (ex: 500 ou timeout)
    with patch.object(bls, "_http_request", return_value=None):
        inds = bls.fetch_latest_indicators()
        assert len(inds) == 0
        assert bls.last_status == "ERROR"

    # Vérification dans la base : la donnée de 2024 est toujours là avec sa date intacte
    stored = mem_db.get_latest_official_indicator("UNEMPLOYMENT", "USD")
    assert stored is not None
    assert stored["reference_period"] == "2024M01"
    assert stored["retrieved_at"] == "2024-02-02T14:00:00+00:00"
    assert stored["status"] == "STALE"


def test_strict_risk_ceiling_enforces_monetary_budget_against_fixed_and_manual_lots():
    """
    Priorité 1 : Vérifie que le plafond strict de risque (strict_risk_ceiling)
    interdit formellement à un lot fixe (ex: 0.20) ou forcé de dépasser le budget
    monétaire autorisé, particulièrement sur petit capital.
    """
    from src.trading.risk_manager import RiskManager

    # Cas A : Compte de 2000 $ avec risque configuré à 1.0% ($20.00) et lot fixe dangereux de 0.20 lot ($80.00)
    rm = RiskManager(
        default_risk_percent=1.0,
        fixed_lot_size=0.20,
        strict_risk_ceiling=True,
    )

    # Calcul du lot pour un trade EURUSD avec 40 pips de SL (perte de $80 pour 0.20 lot)
    final_lot, calc_lot, risk_amt, loss_per_lot = rm.calculate_lot_size(
        capital=2000.0,
        risk_percent=1.0,
        entry_price=1.0850,
        sl_price=1.0810,  # 40 pips -> loss_per_lot = $400/lot
        contract_size=100000.0,
        tick_size=0.00001,
        tick_value=1.0,
        min_lot=0.01,
        max_lot=10.0,
        lot_step=0.01,
    )
    # Le lot fixe (0.20) a été rigoureusement plafonné au lot calculé dynamique (0.05) pour respecter le budget de $20 !
    assert final_lot == 0.05
    assert final_lot * loss_per_lot <= 20.0 * 1.05

    # Cas B : Compte de 200 $ où même le lot minimum broker (0.01 lot = $4.00 de perte) dépasse le budget ($2.00)
    # validate_proposal doit formellement REFUSER le trade en mode strict_risk_ceiling
    rm_micro = RiskManager(default_risk_percent=1.0, strict_risk_ceiling=True)
    forced_proposal = {
        "action": "BUY",
        "entry": 1.0850,
        "sl": 1.0810,
        "tp": 1.0930,
        "confidence": 80.0,
        "lot": 0.01,
        "risk_percent": 1.0,
    }
    account_info = {"balance": 200.0, "equity": 200.0}
    symbol_info = {"trade_tick_value": 1.0, "trade_tick_size": 0.00001, "digits": 5, "volume_min": 0.01, "volume_max": 10.0, "volume_step": 0.01}

    val = rm_micro.validate_proposal(forced_proposal, account_info, symbol_info)
    assert val["valid"] is False
    assert "Plafond Strict de Risque" in val["reason"]


def test_calendar_provider_2027_schedules_and_confidence_distinction(mem_db):
    """
    Priorité 2 : Vérifie la couverture certifiée 2027 du calendrier de secours
    et la distinction stricte entre LIVE_FEED, LOCAL_FALLBACK et UNAVAILABLE.
    """
    from src.economic.calendar_provider import CalendarProvider, VERIFIED_SCHEDULES

    # 1. Présence des calendriers 2027
    assert 2027 in VERIFIED_SCHEDULES["FOMC"]
    assert len(VERIFIED_SCHEDULES["FOMC"][2027]) == 8
    assert 2027 in VERIFIED_SCHEDULES["ECB"]
    assert len(VERIFIED_SCHEDULES["ECB"][2027]) == 8
    assert 2027 in VERIFIED_SCHEDULES["BOE"]
    assert len(VERIFIED_SCHEDULES["BOE"][2027]) == 8
    assert 2027 in VERIFIED_SCHEDULES["BLS_NFP"]
    assert 2027 in VERIFIED_SCHEDULES["BLS_CPI"]

    cal = CalendarProvider(memory=mem_db)

    # 2. Génération des événements pour 2027
    evs_2027 = cal.generate_recurring_fallback_events(target_year=2027, target_month=1)
    fomc_2027 = [e for e in evs_2027 if "FOMC" in e["title"] or "Fed" in e["title"]]
    assert len(fomc_2027) >= 1
    assert fomc_2027[0]["calendar_confidence"] == "VERIFIED"

    # 3. Vérification que hors ligne (sans fetch réseau), is_in_news_blackout ne prétend pas être LIVE_FEED
    blackout_res = cal.is_in_news_blackout("EURUSD")
    assert blackout_res["calendar_status"] in ("LOCAL_FALLBACK", "UNAVAILABLE")
    if not cal.last_fetch_time:
        assert blackout_res["calendar_status"] != "OFFICIAL_LIVE_FEED"


def test_el_economista_macro_fundamentals_and_event_impact_influence_bias(mem_db):
    """
    Priorité 3 : Vérifie que les indicateurs officiels (BLS, BCE, Eurostat)
    et le moteur EventImpactEngine participent directement au calcul du biais final.
    """
    eco = ElEconomista(memory=mem_db)

    # Insertion d'indicateurs macro : chômage USD très bas (3.8%) vs EUR élevé (6.5%)
    mem_db.store_official_indicator({
        "source": "BLS",
        "indicator": "UNEMPLOYMENT",
        "currency_or_region": "USD",
        "value": 3.8,
        "reference_period": "2026M09",
        "status": "OFFICIAL_CURRENT",
    })
    mem_db.store_official_indicator({
        "source": "EUROSTAT",
        "indicator": "UNEMPLOYMENT",
        "currency_or_region": "EUR",
        "value": 6.5,
        "reference_period": "2026-08",
        "status": "OFFICIAL_CURRENT",
    })

    # Évaluation des fondamentaux macro
    macro_fund = eco._evaluate_macro_fundamentals("USD", "EUR")
    assert macro_fund["used_indicators"] >= 1
    assert macro_fund["is_fresh"] is True
    assert macro_fund["bias"] == "BULLISH"  # Favorable au USD face à l'EUR

    # Vérification que le contexte global EURUSD intègre ces fondamentaux
    ctx = eco.analizar_contexto_economico("EURUSD")
    assert "macro_fundamentals" in ctx
    assert ctx["macro_fundamentals"]["used_indicators_count"] >= 1


def test_trade_execution_final_pre_order_send_security_gate():
    """
    Priorité 4 : Audit du verrou ultime précédant mt5.order_send() dans execute_order :
    - Rejet si Stop Loss incohérent (SL >= Prix pour un BUY)
    - Rejet si quota de positions ouvertes atteint
    - Rejet si risque monétaire recalculé dépasse le plafond strict
    """
    mock_connector = MagicMock()
    mock_connector.get_account_info.return_value = {"balance": 1000.0, "equity": 1000.0}
    mock_connector.get_symbol_info.return_value = {
        "ask": 1.0855, "bid": 1.0850, "digits": 5, "trade_tick_value": 1.0, "trade_tick_size": 0.00001
    }

    engine = TradeExecutionEngine(
        mt5_connector=mock_connector,
        mode="auto",
        default_risk_percent=1.0,  # 1.0% de $1000 = $10.00
        max_open_positions=2,
        strict_risk_ceiling=True,
    )

    valid_econ = {
        "status": "OK", "high_impact_window": False, "event_risk": "LOW",
        "recommendation": "NORMAL", "data_freshness": "FRESH", "is_rate_stale": False
    }

    # Cas 1 : Stop Loss incohérent sur un BUY (SL au-dessus du prix d'achat !)
    req_bad_sl = {
        "symbol": "EURUSD", "action": 0, "volume": 0.05,
        "price": 1.0855, "sl": 1.0900, "tp": 1.1000, "type": 0,
        "_economic_context": valid_econ,
    }
    with patch("MetaTrader5.order_send") as mock_send:
        res = engine.execute_order(req_bad_sl)
        assert res["success"] is False
        assert "Stop Loss incohérent" in res["reason"]
        mock_send.assert_not_called()

    # Cas 2 : Quota de positions ouvertes déjà atteint (2 positions)
    mock_connector.get_open_positions.return_value = [{"ticket": 1}, {"ticket": 2}]
    req_quota = {
        "symbol": "EURUSD", "action": 0, "volume": 0.01,
        "price": 1.0855, "sl": 1.0845, "tp": 1.0900, "type": 0,
        "_economic_context": valid_econ,
    }
    with patch("MetaTrader5.order_send") as mock_send:
        res = engine.execute_order(req_quota)
        assert res["success"] is False
        assert "Quota maximal de positions atteint" in res["reason"]
        mock_send.assert_not_called()

    # Cas 3 : Risque excessif violant le plafond strict de risque ($50 de risque vs budget de $10.50)
    mock_connector.get_open_positions.return_value = []
    req_huge_risk = {
        "symbol": "EURUSD", "action": 0, "volume": 0.10,
        "price": 1.0855, "sl": 1.0805, "tp": 1.0950, "type": 0,  # 50 pips * 0.10 lot = $50.00
        "_economic_context": valid_econ,
    }
    with patch("MetaTrader5.order_send") as mock_send:
        res = engine.execute_order(req_huge_risk)
        assert res["success"] is False
        assert "dépasse le plafond strict" in res["reason"]
        mock_send.assert_not_called()


def test_base_provider_health_diagnostics_and_network_reasons():
    """Vérifie que BaseEconomicProvider enregistre le motif exact d'erreur et son bilan de santé."""
    provider = BaseEconomicProvider(name="TEST_PROV", timeout_seconds=1.0, max_retries=1)

    # État initial
    initial_health = provider.get_provider_health()
    assert initial_health["name"] == "TEST_PROV"
    assert initial_health["status"] == "UNINITIALIZED"
    assert initial_health["last_error"] is None

    # Simulation d'un timeout ou erreur de connexion
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timed out")):
        res = provider._http_request("https://fake.economic.api/test")
        assert res is None
        assert provider.last_status == "ERROR"
        assert "timed out" in provider.last_error_reason
        assert provider.failure_count == 1

    # Simulation d'une erreur HTTP 403
    with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("https://fake.url", 403, "Forbidden", {}, None)):
        res = provider._http_request("https://fake.economic.api/test")
        assert res is None
        assert provider.last_http_code == 403
        assert "HTTP 403" in provider.last_error_reason
        assert provider.failure_count == 2

    health = provider.get_provider_health()
    assert health["status"] == "ERROR"
    assert health["failure_count"] == 2
    assert health["last_http_code"] == 403


def test_el_economista_generar_tabla_salud_economica(mem_db):
    """Vérifie la génération et l'affichage complet du Tableau de santé économique."""
    now_str = datetime.now(timezone.utc).isoformat()
    # Insertion d'indicateurs de test
    mem_db.store_official_indicator({
        "source": "BLS",
        "indicator": "UNEMPLOYMENT",
        "currency_or_region": "USD",
        "value": 4.1,
        "unit": "PERCENT",
        "reference_period": "2026M09",
        "published_at": now_str,
        "retrieved_at": now_str,
        "status": "OFFICIAL_CURRENT",
    })
    mem_db.store_official_indicator({
        "source": "ECB",
        "indicator": "HICP_INFLATION",
        "currency_or_region": "EUR",
        "value": 1.9,
        "unit": "PERCENT",
        "reference_period": "2025-12",
        "published_at": now_str,
        "retrieved_at": now_str,
        "status": "STALE",
    })

    eco = ElEconomista(memory=mem_db)
    tabla = eco.generar_tabla_salud_economica()

    assert "TABLEAU DE SANTE ECONOMIQUE" in tabla
    assert "BLS (Etats-Unis)" in tabla
    assert "BCE (Zone euro)" in tabla
    assert "UNEMPLOYMENT" in tabla
    assert "4.1 PERCENT" in tabla
    assert "HICP_INFLATION" in tabla
    assert "[STALE]" in tabla
    assert "CALENDRIER ECONOMIQUE" in tabla

    # Vérification que l'impression console s'exécute sans UnicodeEncodeError
    eco.imprimir_tabla_salud_economica()



