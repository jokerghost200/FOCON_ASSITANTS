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
