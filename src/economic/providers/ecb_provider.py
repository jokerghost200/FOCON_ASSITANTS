# -*- coding: utf-8 -*-
"""
FATI ECB Provider (Connecteur Public Banque Centrale Européenne - SDMX REST API)
================================================================================
Récupère les séries officielles de la zone euro via le portail de données officiel de la BCE :
- FM/B.U2.EUR.4F.KR.DFR.LEV : Taux de la facilité de dépôt (Deposit Facility Rate)
- FM/B.U2.EUR.4F.KR.MRR_FR.LEV : Taux des opérations principales de refinancement (MRO Fixed Rate)
- FM/B.U2.EUR.4F.KR.MLFR.LEV : Taux de la facilité de prêt marginal (Marginal Lending Facility Rate)
- ICP/M.U2.N.000000.4.ANR : Inflation IPCH globale annuelle de la zone euro (HICP Annual Rate)

Caractéristiques :
- Format officiel standard SDMX 1.0 JSON (aucun abonnement ni clé API).
- Extraction précise de l'horodatage officiel et de la valeur.
- Enregistrement immédiat dans EconomicMemory.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from src.economic.providers.base_provider import BaseEconomicProvider
from src.economic.economic_memory import EconomicMemory

logger = logging.getLogger(__name__)

ECB_BASE_URL = "https://data-api.ecb.europa.eu/service/data"

ECB_SERIES_CONFIG = [
    {
        "flow": "FM",
        "key": "B.U2.EUR.4F.KR.DFR.LEV",
        "indicator": "DEPOSIT_FACILITY_RATE",
        "unit": "PERCENT",
        "description": "ECB Deposit Facility Rate",
    },
    {
        "flow": "FM",
        "key": "B.U2.EUR.4F.KR.MRR_FR.LEV",
        "indicator": "MAIN_REFINANCING_RATE",
        "unit": "PERCENT",
        "description": "ECB Main Refinancing Operations Rate",
    },
    {
        "flow": "FM",
        "key": "B.U2.EUR.4F.KR.MLFR.LEV",
        "indicator": "MARGINAL_LENDING_RATE",
        "unit": "PERCENT",
        "description": "ECB Marginal Lending Facility Rate",
    },
    {
        "flow": "ICP",
        "key": "M.U2.N.000000.4.ANR",
        "indicator": "HICP_INFLATION",
        "unit": "PERCENT",
        "description": "Euro Area HICP Annual Rate of Change",
    },
]


class ECBProvider(BaseEconomicProvider):
    """
    Connecteur officiel pour la Banque Centrale Européenne (BCE SDMX REST).
    """

    def __init__(
        self,
        memory: Optional[EconomicMemory] = None,
        timeout_seconds: float = 10.0,
    ):
        super().__init__(name="ECB", timeout_seconds=timeout_seconds, max_retries=2)
        self.memory = memory or EconomicMemory()

    def fetch_latest_indicators(self, last_n_observations: int = 2) -> List[Dict[str, Any]]:
        """
        Interroge l'API SDMX de la BCE pour chaque série configurée.
        Retourne les indicateurs normalisés et les stocke dans EconomicMemory.
        """
        collected_indicators: List[Dict[str, Any]] = []

        for cfg in ECB_SERIES_CONFIG:
            flow = cfg["flow"]
            key = cfg["key"]
            url = f"{ECB_BASE_URL}/{flow}/{key}?lastNObservations={last_n_observations}&format=jsondata"

            raw_bytes = self._http_request(url, headers={"Accept": "application/json"})
            data = self._safe_json_loads(raw_bytes)
            if not data:
                logger.warning(f"🛑 [ECBProvider] Données indisponibles pour la série {key}")
                continue

            # Parsing des dimensions temporelles SDMX
            time_values = (
                data.get("structure", {})
                .get("dimensions", {})
                .get("observation", [{}])[0]
                .get("values", [])
            )
            time_labels = [t.get("id") or str(t.get("name")) for t in time_values]

            series_dict = data.get("dataSets", [{}])[0].get("series", {})
            for _, s_info in series_dict.items():
                observations = s_info.get("observations", {})
                # Trier par index numérique croissant pour trouver la dernière observation
                sorted_obs_keys = sorted(observations.keys(), key=lambda x: int(x))
                if not sorted_obs_keys:
                    continue

                latest_idx_str = sorted_obs_keys[-1]
                latest_idx = int(latest_idx_str)
                obs_val_list = observations[latest_idx_str]

                if not obs_val_list or obs_val_list[0] is None:
                    continue

                try:
                    val_float = float(obs_val_list[0])
                except (ValueError, TypeError):
                    continue

                ref_period = time_labels[latest_idx] if latest_idx < len(time_labels) else "LATEST"

                # Date de publication estimée ou effective
                published_iso = f"{ref_period}T00:00:00+00:00" if len(ref_period) == 10 else None

                ind = self.normalize_indicator(
                    indicator=cfg["indicator"],
                    currency_or_region="EUR",
                    value=val_float,
                    unit=cfg["unit"],
                    reference_period=ref_period,
                    source_url=url,
                    published_at=published_iso,
                    status="OFFICIAL_CURRENT",
                    is_official=True,
                    revision="INITIAL",
                )
                collected_indicators.append(ind)
                self.memory.store_official_indicator(ind)

        logger.info(f"🏦 [ECBProvider] {len(collected_indicators)} indicateurs BCE récupérés et validés.")
        return collected_indicators
