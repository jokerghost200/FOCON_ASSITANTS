# -*- coding: utf-8 -*-
"""
FATI Eurostat Provider (Connecteur Public Eurostat - Commission Européenne)
==========================================================================
Récupère les séries macroéconomiques officielles de la zone euro via l'API statistique Eurostat :
- prc_hicp_manr : Inflation IPCH globale mensuelle (taux de variation annuel)
- une_rt_m : Taux de chômage mensuel harmonisé (s_adj=SA, age=TOTAL)
- namq_10_gdp : Croissance du PIB trimestriel (B1GQ, unit=CLV_PCH_SM, s_adj=SCA)

Caractéristiques :
- Gratuit, public, sans authentification obligatoire.
- Résolution dynamique de la zone euro (EA21 / EA20 / EA).
- Filtrage des périodes futures non encore publiées.
- Stockage direct dans EconomicMemory.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from src.economic.providers.base_provider import BaseEconomicProvider
from src.economic.economic_memory import EconomicMemory

logger = logging.getLogger(__name__)

EUROSTAT_BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

EUROSTAT_DATASETS = [
    {
        # Remplacement officiel du jeu prc_hicp_manr arrêté
        "code": "prc_hicp_minr",
        "indicator": "HICP_INFLATION",
        "unit": "PERCENT",
        "params": "unit=RCH_A&coicop18=TOTAL&lastTimePeriod=6",
        "geo_candidates": ["EA21", "EA20", "EA"],
        "fallback_params": ["unit=RCH_A&coicop=CP00&lastTimePeriod=6"],
        "legacy_code": "prc_hicp_manr",
        "legacy_params": "coicop=CP00&lastTimePeriod=6",
        "description": "Euro Area HICP Monthly Inflation (Annual Rate, replacement prc_hicp_minr)",
    },
    {
        "code": "une_rt_m",
        "indicator": "UNEMPLOYMENT",
        "unit": "PERCENT",
        "params": "s_adj=SA&age=TOTAL&unit=PC_ACT&sex=T&lastTimePeriod=6",
        "geo_candidates": ["EA21", "EA20", "EA"],
        "description": "Euro Area Harmonized Unemployment Rate",
    },
    {
        "code": "namq_10_gdp",
        "indicator": "GDP_GROWTH_YOY",
        "unit": "PERCENT",
        "params": "unit=CLV_PCH_SM&na_item=B1GQ&s_adj=SCA&lastTimePeriod=6",
        "geo_candidates": ["EA21", "EA20", "EA"],
        "description": "Euro Area Real GDP Growth Rate (YoY)",
    },
]


class EurostatProvider(BaseEconomicProvider):
    """
    Connecteur officiel pour les statistiques macroéconomiques d'Eurostat.
    """
    expected_indicators = [
        "HICP_INFLATION",
        "UNEMPLOYMENT",
        "GDP_GROWTH_YOY",
    ]

    def __init__(
        self,
        memory: Optional[EconomicMemory] = None,
        timeout_seconds: float = 10.0,
    ):
        super().__init__(name="EUROSTAT", timeout_seconds=timeout_seconds, max_retries=2)
        self.memory = memory or EconomicMemory()

    def fetch_latest_indicators(self) -> List[Dict[str, Any]]:
        """
        Interroge Eurostat pour chaque indicateur configuré.
        Supporte les jeux actifs (prc_hicp_minr avec coicop18=TOTAL) et fallbacks documentés.
        Valide rigoureusement l'appartenance à la zone euro (EA21/EA20/EA).
        Retourne la liste des indicateurs normalisés et les stocke dans EconomicMemory.
        """
        collected_indicators: List[Dict[str, Any]] = []

        for ds in EUROSTAT_DATASETS:
            code = ds["code"]
            indicator_name = ds["indicator"]
            unit = ds["unit"]
            primary_params = ds["params"]
            geo_candidates = ds["geo_candidates"]

            # Préparer les variantes : flux primaire, puis fallbacks si présents
            query_variants = [(code, primary_params)]
            for fb_p in ds.get("fallback_params", []):
                query_variants.append((code, fb_p))
            if "legacy_code" in ds and "legacy_params" in ds:
                query_variants.append((ds["legacy_code"], ds["legacy_params"]))

            dataset_success = False

            for query_code, query_params in query_variants:
                if dataset_success:
                    break

                for geo in geo_candidates:
                    url = f"{EUROSTAT_BASE_URL}/{query_code}?geo={geo}&{query_params}"
                    raw_bytes = self._http_request(url, headers={"Accept": "application/json"})
                    if raw_bytes is None:
                        continue

                    data = self._safe_json_loads(raw_bytes)
                    if not data:
                        continue

                    if "dimension" not in data or not isinstance(data.get("dimension"), dict):
                        self.record_error("SCHEMA_VALIDATION", f"Structure Eurostat inattendue pour {query_code}: 'dimension' manquante")
                        logger.warning(f"🛑 [EurostatProvider] Structure inattendue pour {query_code}")
                        continue

                    # Validation géographique stricte : la réponse DOIT cibler la zone euro
                    geo_dim = data.get("dimension", {}).get("geo", {}).get("category", {})
                    returned_geos = set(geo_dim.get("index", {}).keys())
                    valid_ea_codes = {"EA", "EA21", "EA20", "EA19"}
                    if returned_geos and not returned_geos.intersection(valid_ea_codes):
                        reason = f"Zone géographique non Eurozone reçue ({returned_geos}) pour {query_code}"
                        self.record_error("SCHEMA_VALIDATION", reason)
                        logger.warning(f"🛑 [EurostatProvider] ⚠️ {reason}")
                        continue

                    values_dict = data.get("value", {})
                    if not values_dict:
                        # Données vides pour ce code geo, tente le suivant
                        continue

                    category = data.get("dimension", {}).get("time", {}).get("category", {})
                    time_index = category.get("index", {})
                    time_label = category.get("label", {})

                    # Récupérer la dernière observation valide disponible
                    sorted_indices = sorted(values_dict.keys(), key=lambda x: int(x))
                    if not sorted_indices:
                        continue

                    latest_idx_str = sorted_indices[-1]
                    latest_idx = int(latest_idx_str)
                    latest_val = values_dict[latest_idx_str]

                    try:
                        val_float = float(latest_val)
                    except (ValueError, TypeError):
                        continue

                    # Association stricte et mathématique de la période à l'index de valeur
                    if time_index:
                        idx_to_time = {int(v): k for k, v in time_index.items()}
                        time_code = idx_to_time.get(latest_idx)
                        ref_period = str(time_label.get(time_code, time_code)) if time_code else "LATEST"
                    else:
                        time_list = list(time_label.values())
                        ref_period = str(time_list[latest_idx]) if latest_idx < len(time_list) else "LATEST"

                    ind = self.normalize_indicator(
                        indicator=indicator_name,
                        currency_or_region="EUR",
                        value=val_float,
                        unit=unit,
                        reference_period=ref_period,
                        published_at=None,
                        source_url=url,
                        is_official=True,
                        revision="INITIAL",
                    )
                    collected_indicators.append(ind)
                    self.memory.store_official_indicator(ind)
                    dataset_success = True
                    break

            if not dataset_success:
                logger.warning(f"🛑 [EurostatProvider] Impossible d'obtenir {indicator_name} depuis Eurostat")

        if collected_indicators:
            self.record_business_success(len(collected_indicators))
            logger.info(f"🇪🇺 [EurostatProvider] {len(collected_indicators)} indicateurs Eurostat récupérés et validés.")
        else:
            if not self.last_error_reason:
                self.record_error("SCHEMA_VALIDATION", "Aucun indicateur Eurostat valide extrait")

        return collected_indicators
