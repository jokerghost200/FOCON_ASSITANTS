# -*- coding: utf-8 -*-
"""
FATI BLS Provider (Connecteur Public Bureau of Labor Statistics - US)
====================================================================
Récupère les séries officielles américaines via l'API v1 publique gratuite (sans clé API) :
- LNS14000000 : Taux de chômage américain (Civilian Unemployment Rate)
- CUUR0000SA0 : Indice des prix à la consommation (CPI-U All items) + Calcul YoY
- CES0000000001 : Emploi non-agricole salarié (Total Nonfarm Payrolls)

Règles de sécurité et robustesse :
- Respect des quotas publics v1 gratuits (cache local et requêtes espacées).
- Filtrage des valeurs non numériques (ex: '-' lors d'arrêts administratifs temporaires).
- Enregistrement immédiat dans EconomicMemory avec traçabilité complète.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from src.economic.providers.base_provider import BaseEconomicProvider
from src.economic.economic_memory import EconomicMemory

logger = logging.getLogger(__name__)

BLS_API_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"

SERIES_METADATA = {
    "LNS14000000": {
        "indicator": "UNEMPLOYMENT",
        "unit": "PERCENT",
        "description": "US Civilian Unemployment Rate (Seasonally Adjusted)",
    },
    "CUUR0000SA0": {
        "indicator": "CPI_INDEX",
        "unit": "INDEX_POINTS",
        "description": "US Consumer Price Index for All Urban Consumers (Unadjusted)",
    },
    "CES0000000001": {
        "indicator": "NONFARM_PAYROLLS",
        "unit": "THOUSANDS",
        "description": "US Total Nonfarm Employment",
    },
}


class BLSProvider(BaseEconomicProvider):
    """
    Connecteur officiel pour le Bureau of Labor Statistics (États-Unis).
    """

    def __init__(
        self,
        memory: Optional[EconomicMemory] = None,
        timeout_seconds: float = 10.0,
    ):
        super().__init__(name="BLS", timeout_seconds=timeout_seconds, max_retries=2)
        self.memory = memory or EconomicMemory()

    def fetch_latest_indicators(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Interroge l'API v1 BLS pour les séries prioritaires US.
        Retourne la liste des indicateurs normalisés et les stocke dans EconomicMemory.
        """
        now = datetime.now(timezone.utc)
        cur_year = end_year or now.year
        # Récupère l'année courante et les 2 années précédentes pour couvrir au minimum 24 mois
        st_year = start_year or (cur_year - 2)

        payload = {
            "seriesid": list(SERIES_METADATA.keys()),
            "startyear": str(st_year),
            "endyear": str(cur_year),
        }
        payload_bytes = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        raw_bytes = self._http_request(
            BLS_API_URL,
            method="POST",
            headers=headers,
            data=payload_bytes,
        )

        data = self._safe_json_loads(raw_bytes)
        if not data or data.get("status") != "REQUEST_SUCCEEDED":
            self.last_status = "ERROR"
            msg = data.get("message") if data else "Aucune réponse reçue"
            logger.warning(f"🛑 [BLSProvider] Échec récupération BLS: {msg}")
            return []

        series_list = data.get("Results", {}).get("series", [])
        collected_indicators: List[Dict[str, Any]] = []

        for s in series_list:
            s_id = s.get("seriesID")
            meta = SERIES_METADATA.get(s_id)
            if not meta:
                continue

            data_points = s.get("data", [])
            valid_points = []
            for dp in data_points:
                val_str = dp.get("value", "").strip()
                try:
                    val_float = float(val_str)
                    valid_points.append({
                        "year": dp.get("year"),
                        "period": dp.get("period"),
                        "periodName": dp.get("periodName"),
                        "value": val_float,
                    })
                except ValueError:
                    # Ignore les valeurs indisponibles (ex: '-' lors de fermeture gouvernementale)
                    continue

            if not valid_points:
                continue

            latest_dp = valid_points[0]
            ref_period = f"{latest_dp['year']}{latest_dp['period']}"

            ind = self.normalize_indicator(
                indicator=meta["indicator"],
                currency_or_region="USD",
                value=latest_dp["value"],
                unit=meta["unit"],
                reference_period=ref_period,
                source_url="https://api.bls.gov/publicAPI/v1/timeseries/data/",
                published_at=None,
                is_official=True,
                revision="INITIAL",
            )
            collected_indicators.append(ind)
            self.memory.store_official_indicator(ind)

            # Si c'est l'indice CPI et qu'on dispose de 12 mois d'historique, calcul de l'inflation annuelle (YoY)
            if meta["indicator"] == "CPI_INDEX" and len(valid_points) >= 13:
                # Chercher le point d'il y a 1 an même période (ex: M12)
                same_period = [p for p in valid_points[1:] if p["period"] == latest_dp["period"]]
                if same_period:
                    prev_year_point = same_period[0]
                    prev_cpi = prev_year_point["value"]
                    if prev_cpi > 0:
                        yoy_rate = round(((latest_dp["value"] - prev_cpi) / prev_cpi) * 100.0, 2)
                        cpi_yoy = self.normalize_indicator(
                            indicator="CPI_YOY",
                            currency_or_region="USD",
                            value=yoy_rate,
                            unit="PERCENT",
                            reference_period=ref_period,
                            source_url="https://api.bls.gov/publicAPI/v1/timeseries/data/",
                            published_at=None,
                            is_official=True,
                            revision="INITIAL",
                        )
                        collected_indicators.append(cpi_yoy)
                        self.memory.store_official_indicator(cpi_yoy)

        logger.info(f"📊 [BLSProvider] {len(collected_indicators)} indicateurs US récupérés et validés.")
        return collected_indicators
