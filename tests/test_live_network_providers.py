# -*- coding: utf-8 -*-
"""
Test d'Intégration Réseau Réel (Live Network & Freshness Audit) pour El Economista
==================================================================================
Interroge en conditions réelles les serveurs publics distants (BLS, BCE, Eurostat) :
- Vérifie la joignabilité effective des URL officielles.
- Extrait les dernières observations disponibles en temps réel (sans forcer d'années passées).
- Valide la séparation rigoureuse :
    * reference_period (période observée)
    * effective_date (date d'effet officiel du taux)
    * published_at (date réelle de publication, None si non fournie)
    * retrieved_at (horodatage UTC de capture)
- Évalue objectivement le statut économique : OFFICIAL_CURRENT vs STALE.
- Sépare le contrôle de fraîcheur temps réel du test historique reproductible.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from src.economic.economic_memory import EconomicMemory
from src.economic.providers.bls_provider import BLSProvider, BLS_API_URL
from src.economic.providers.ecb_provider import ECBProvider, ECB_BASE_URL
from src.economic.providers.eurostat_provider import EurostatProvider, EUROSTAT_BASE_URL
from src.cerebros.el_economista import ElEconomista


def test_live_network_official_providers():
    """
    Test 1 (Live Freshness) : Interroge la dernière période réellement disponible en direct.
    Vérifie l'actualité des données, la séparation des dates et l'évaluation du statut économique.
    """
    mem = EconomicMemory(db_path=":memory:")
    bls = BLSProvider(memory=mem, timeout_seconds=15.0)
    ecb = ECBProvider(memory=mem, timeout_seconds=15.0)
    eurostat = EurostatProvider(memory=mem, timeout_seconds=15.0)

    print("\n" + "=" * 85)
    print(">> CONTROLE DE FRAICHEUR ET INTEGRATION RESEAU DIRECT -- FOCON ASSISTANT")
    print("=" * 85)

    # ── 1. VÉRIFICATION EN DIRECT : BLS (Bureau of Labor Statistics) ──
    # Interrogation dynamique sans fixer d'année passée (cherche l'année en cours et précédentes)
    bls_indicators = bls.fetch_latest_indicators()
    assert len(bls_indicators) > 0, "Echec de collecte reseau en direct aupres du BLS"
    assert bls.last_status == "SUCCESS"

    unemp_ind = next((i for i in bls_indicators if i["indicator"] == "UNEMPLOYMENT"), bls_indicators[0])
    cpi_yoy_ind = next((i for i in bls_indicators if i["indicator"] == "CPI_YOY"), None)

    current_year = datetime.now(timezone.utc).year
    # Vérification que la période obtenue est contemporaine (année courante ou précédente pour début d'année)
    assert (str(current_year) in str(unemp_ind["reference_period"]) or str(current_year - 1) in str(unemp_ind["reference_period"])), (
        f"Période BLS attendue contemporaine ({current_year}), reçue: {unemp_ind['reference_period']}"
    )
    assert unemp_ind["age_days"] <= 75.0, f"L'indicateur chômage doit avoir un âge <= 75j, reçu: {unemp_ind.get('age_days')}j"
    assert unemp_ind["status"] == "OFFICIAL_CURRENT"
    assert unemp_ind["value"] > 0.0
    assert unemp_ind["published_at"] is None, "BLS v1 ne fournissant pas published_at, la valeur doit rester None"

    print("\n[SOURCE 1] BLS -- Etats-Unis (Bureau of Labor Statistics)")
    print(f"  - Statut HTTP             : {bls.last_status} (Public v1 joignable)")
    print(f"  - URL reellement appelee  : {BLS_API_URL}")
    print(f"  - Chomage US observe      : {unemp_ind['value']} {unemp_ind['unit']}")
    if cpi_yoy_ind:
        print(f"  - CPI YoY calcule (reel)  : {cpi_yoy_ind['value']} {cpi_yoy_ind['unit']}")
    print(f"  - Periode de reference    : {unemp_ind['reference_period']}")
    print(f"  - Date de publication     : {unemp_ind.get('published_at')} (Non fournie -> None explicite)")
    print(f"  - Horodatage collecte UTC : {unemp_ind['retrieved_at']}")
    print(f"  - Age economique calcule  : {unemp_ind.get('age_days')} jours -> Statut: {unemp_ind['status']}")

    # ── 2. VÉRIFICATION EN DIRECT : BCE (Banque Centrale Européenne - SDMX) ──
    ecb_indicators = ecb.fetch_latest_indicators(last_n_observations=2)
    assert len(ecb_indicators) > 0, "Echec de collecte reseau en direct aupres de la BCE SDMX"
    assert ecb.last_status == "SUCCESS"

    dfr_ind = next((i for i in ecb_indicators if i["indicator"] == "DEPOSIT_FACILITY_RATE"), ecb_indicators[0])
    hicp_ind = next((i for i in ecb_indicators if i["indicator"] == "HICP_INFLATION"), None)

    # Vérification de la distinction stricte effective_date vs published_at
    assert dfr_ind["effective_date"] is not None, "Le taux directeur BCE doit comporter une date d'effet (effective_date)"
    assert dfr_ind["published_at"] is None, "La date de publication ne doit pas etre fabriquee a partir de la date d'effet"
    assert dfr_ind["value"] > 0.0

    print("\n[SOURCE 2] BCE -- Zone Euro (Banque Centrale Europeenne - SDMX)")
    print(f"  - Statut SDMX             : {ecb.last_status} (REST SDMX officiel)")
    print(f"  - URL reellement appelee  : {dfr_ind['source_url']}")
    print(f"  - Taux facilite depot DFR : {dfr_ind['value']} {dfr_ind['unit']}")
    if hicp_ind:
        print(f"  - Inflation IPCH annuelle : {hicp_ind['value']} {hicp_ind['unit']} (Statut: {hicp_ind['status']}, Age: {hicp_ind.get('age_days')}j)")
    print(f"  - Periode de reference    : {dfr_ind['reference_period']}")
    print(f"  - Date d'effet officiel   : {dfr_ind['effective_date']}")
    print(f"  - Date de publication     : {dfr_ind.get('published_at')} (Non fournie dans FM -> None explicite)")
    print(f"  - Horodatage collecte UTC : {dfr_ind['retrieved_at']}")
    print(f"  - Age economique calcule  : {dfr_ind.get('age_days')} jours -> Statut: {dfr_ind['status']}")

    # ── 3. VÉRIFICATION EN DIRECT : Eurostat (Commission Européenne) ──
    eurostat_indicators = eurostat.fetch_latest_indicators()
    assert len(eurostat_indicators) > 0, "Echec de collecte reseau en direct aupres d'Eurostat"
    assert eurostat.last_status == "SUCCESS"

    hicp_eu = next((i for i in eurostat_indicators if i["indicator"] == "HICP_INFLATION"), eurostat_indicators[0])
    une_eu = next((i for i in eurostat_indicators if i["indicator"] == "UNEMPLOYMENT"), None)
    gdp_eu = next((i for i in eurostat_indicators if i["indicator"] == "GDP_GROWTH_YOY"), None)

    print("\n[SOURCE 3] Eurostat -- Zone Euro (Statistiques Europeennes)")
    print(f"  - Statut API              : {eurostat.last_status} (Diffusion publique)")
    print(f"  - URL reellement appelee  : {hicp_eu['source_url']}")
    print(f"  - Inflation IPCH Zone Euro: {hicp_eu['value']} {hicp_eu['unit']} (Periode: {hicp_eu['reference_period']}, Statut: {hicp_eu['status']})")
    if une_eu:
        print(f"  - Chomage harmonise EA    : {une_eu['value']} {une_eu['unit']} (Periode: {une_eu['reference_period']}, Statut: {une_eu['status']})")
    if gdp_eu:
        print(f"  - Croissance PIB reel YoY : {gdp_eu['value']} {gdp_eu['unit']} (Periode: {gdp_eu['reference_period']}, Statut: {gdp_eu['status']})")
    print(f"  - Horodatage collecte UTC : {hicp_eu['retrieved_at']}")

    # ── 4. CONTRÔLE DE SYNTHÈSE EL ECONOMISTA ET FAIL-CLOSED ──
    eco = ElEconomista(memory=mem, bls_provider=bls, ecb_provider=ecb, eurostat_provider=eurostat)
    context = eco.analizar_contexto_economico("EURUSD")

    print("\n[SYNTHESE] El Economista -- Contexte Macro et Fraicheur pour EURUSD")
    print(f"  - Biais global calcule    : {context['bias']} (Confiance : {context['confidence']})")
    print(f"  - Recommandation          : {context['recommendation']}")
    print(f"  - News Blackout (Gel)     : {context['high_impact_window']} (Risque : {context['event_risk']})")
    print(f"  - Sentiment actualites    : {context['news_sentiment']} ({context['news_sentiment_details']['pair_sentiment_bias']})")
    print(f"  - Fraicheur globale       : {context['data_freshness']}")
    print(f"  - Fraicheur detaillee     : {context['sources_freshness']}")
    print("=" * 85 + "\n")

    assert context["status"] == "OK"
    assert "sources_freshness" in context
    assert "confidence" in context["sources_freshness"]["calendar"]
    assert "news_sentiment_details" in context


def test_bls_historical_reproducibility():
    """
    Test 2 (Historical Reproducibility) : Interroge une fenêtre passée fixe (2024-2025)
    pour valider la reproductibilité mathématique des séries et du calcul d'inflation annuelle.
    """
    mem = EconomicMemory(db_path=":memory:")
    bls = BLSProvider(memory=mem, timeout_seconds=15.0)

    # Requête explicitement bornée aux années 2024-2025
    hist_indicators = bls.fetch_latest_indicators(start_year=2024, end_year=2025)
    assert len(hist_indicators) > 0
    assert bls.last_status == "SUCCESS"

    unemp = next((i for i in hist_indicators if i["indicator"] == "UNEMPLOYMENT"), None)
    assert unemp is not None
    assert "2025" in str(unemp["reference_period"]) or "2024" in str(unemp["reference_period"])
    # Donnée de 2024/2025 évaluée en 2026 -> Doit être classée STALE
    assert unemp["status"] == "STALE"
    assert unemp["age_days"] > 90.0
