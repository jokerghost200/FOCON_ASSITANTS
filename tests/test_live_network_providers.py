# -*- coding: utf-8 -*-
"""
Test d'Intégration Réseau Réel (Live Verification) pour El Economista
=====================================================================
Interroge en conditions réelles les serveurs publics distants (BLS, BCE, Eurostat) :
- Vérifie la joignabilité effective des URL officielles
- Extrait les valeurs numériques brutes directement des flux HTTP
- Affiche et certifie la date de publication, la période de référence et l'horodatage UTC de capture
- Démontre l'intégration complète dans EconomicMemory et ElEconomista
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from src.economic.economic_memory import EconomicMemory
from src.economic.providers.bls_provider import BLSProvider, BLS_API_URL
from src.economic.providers.ecb_provider import ECBProvider, ECB_BASE_URL
from src.economic.providers.eurostat_provider import EurostatProvider, EUROSTAT_BASE_URL
from src.cerebros.el_economista import ElEconomista


@pytest.mark.integration
def test_live_network_official_providers():
    """
    Test d'intégration réseau direct : interroge BLS, BCE et Eurostat sur Internet.
    """
    mem = EconomicMemory(db_path=":memory:")
    bls = BLSProvider(memory=mem, timeout_seconds=15.0)
    ecb = ECBProvider(memory=mem, timeout_seconds=15.0)
    eurostat = EurostatProvider(memory=mem, timeout_seconds=15.0)

    print("\n" + "=" * 80)
    print(">> DEBUT DU CONTROLE D'INTEGRATION RESEAU REEL -- FOCON ASSISTANT")
    print("=" * 80)

    # ── 1. VÉRIFICATION EN DIRECT : BLS (Bureau of Labor Statistics) ──
    bls_indicators = bls.fetch_latest_indicators(start_year=2024, end_year=2025)
    assert len(bls_indicators) > 0, "Échec de collecte réseau en direct auprès du BLS"
    assert bls.last_status == "SUCCESS"

    unemp_ind = next((i for i in bls_indicators if i["indicator"] == "UNEMPLOYMENT"), bls_indicators[0])
    cpi_yoy_ind = next((i for i in bls_indicators if i["indicator"] == "CPI_YOY"), None)

    print("\n[SOURCE 1] BLS -- Etats-Unis (Bureau of Labor Statistics)")
    print(f"  - Statut de connexion     : {bls.last_status} (Serveur joignable sans cle API)")
    print(f"  - Source officielle       : {unemp_ind['source']}")
    print(f"  - URL reellement appelee  : {BLS_API_URL}")
    print(f"  - Valeur reelle chomage   : {unemp_ind['value']} {unemp_ind['unit']}")
    if cpi_yoy_ind:
        print(f"  - Valeur reelle CPI YoY   : {cpi_yoy_ind['value']} {cpi_yoy_ind['unit']}")
    print(f"  - Periode de reference    : {unemp_ind['reference_period']}")
    print(f"  - Date de publication     : {unemp_ind.get('published_at') or 'Publie pour la periode'}")
    print(f"  - Horodatage collecte UTC : {unemp_ind['retrieved_at']}")

    # ── 2. VÉRIFICATION EN DIRECT : BCE (Banque Centrale Européenne) ──
    ecb_indicators = ecb.fetch_latest_indicators(last_n_observations=2)
    assert len(ecb_indicators) > 0, "Échec de collecte réseau en direct auprès de la BCE SDMX"
    assert ecb.last_status == "SUCCESS"

    dfr_ind = next((i for i in ecb_indicators if i["indicator"] == "DEPOSIT_FACILITY_RATE"), ecb_indicators[0])
    hicp_ind = next((i for i in ecb_indicators if i["indicator"] == "HICP_INFLATION"), None)

    print("\n[SOURCE 2] BCE -- Zone Euro (Banque Centrale Europeenne - SDMX)")
    print(f"  - Statut de connexion     : {ecb.last_status} (Service SDMX officiel)")
    print(f"  - Source officielle       : {dfr_ind['source']}")
    print(f"  - URL reellement appelee  : {dfr_ind['source_url']}")
    print(f"  - Taux facilite depot DFR : {dfr_ind['value']} {dfr_ind['unit']}")
    if hicp_ind:
        print(f"  - Inflation IPCH annuelle : {hicp_ind['value']} {hicp_ind['unit']}")
    print(f"  - Periode / Date d'effet  : {dfr_ind['reference_period']}")
    print(f"  - Date de publication ISO : {dfr_ind.get('published_at')}")
    print(f"  - Horodatage collecte UTC : {dfr_ind['retrieved_at']}")

    # ── 3. VÉRIFICATION EN DIRECT : Eurostat (Commission Européenne) ──
    eurostat_indicators = eurostat.fetch_latest_indicators()
    assert len(eurostat_indicators) > 0, "Échec de collecte réseau en direct auprès d'Eurostat"
    assert eurostat.last_status == "SUCCESS"

    hicp_eu = next((i for i in eurostat_indicators if i["indicator"] == "HICP_INFLATION"), eurostat_indicators[0])
    une_eu = next((i for i in eurostat_indicators if i["indicator"] == "UNEMPLOYMENT"), None)

    print("\n[SOURCE 3] Eurostat -- Zone Euro (Statistiques Europeennes)")
    print(f"  - Statut de connexion     : {eurostat.last_status} (API statistique officielle)")
    print(f"  - Source officielle       : {hicp_eu['source']}")
    print(f"  - URL reellement appelee  : {hicp_eu['source_url']}")
    print(f"  - Inflation IPCH Zone Euro: {hicp_eu['value']} {hicp_eu['unit']}")
    if une_eu:
        print(f"  - Chomage harmonise EA    : {une_eu['value']} {une_eu['unit']}")
    print(f"  - Periode de reference    : {hicp_eu['reference_period']}")
    print(f"  - Horodatage collecte UTC : {hicp_eu['retrieved_at']}")

    # ── 4. VÉRIFICATION DE LA SYNTHÈSE DANS EL ECONOMISTA ──
    eco = ElEconomista(memory=mem, bls_provider=bls, ecb_provider=ecb, eurostat_provider=eurostat)
    context = eco.analizar_contexto_economico("EURUSD")

    print("\n[SYNTHESE] El Economista -- Contexte Analyse pour EURUSD")
    print(f"  - Biais global calcule    : {context['bias']} (Confiance : {context['confidence']})")
    print(f"  - Recommandation          : {context['recommendation']}")
    print(f"  - News Blackout (Gel)     : {context['high_impact_window']} (Risque : {context['event_risk']})")
    print(f"  - Sentiment actualites    : {context['news_sentiment']} ({context['news_sentiment_details']['pair_sentiment_bias']})")
    print(f"  - Fraicheur globale       : {context['data_freshness']}")
    print(f"  - Fraicheur detaillee     : {context['sources_freshness']}")
    print("=" * 80 + "\n")

    assert context["status"] == "OK"
    assert "sources_freshness" in context
    assert "news_sentiment_details" in context
