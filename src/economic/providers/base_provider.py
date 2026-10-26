# -*- coding: utf-8 -*-
"""
FATI Base Economic Provider (Connecteur Source Économique Abstrait)
==================================================================
Fournit l'infrastructure réseau résiliente pour les connecteurs publics gratuits :
- Gestion de timeouts et tentatives avec backoff exponentiel
- Contexte SSL sécurisé avec fallback automatique certifi
- Validation stricte des données et structures JSON/XML
- Traçabilité et formatage standardisé des indicateurs officiels
"""
from __future__ import annotations

import time
import ssl
import json
import logging
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class BaseEconomicProvider:
    """
    Classe de base pour tous les connecteurs de données macroéconomiques publiques.
    """

    def __init__(
        self,
        name: str,
        timeout_seconds: float = 8.0,
        max_retries: int = 2,
        backoff_factor: float = 1.0,
    ):
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.last_fetch_time: Optional[str] = None
        self.last_status: str = "UNINITIALIZED"

    def _get_ssl_context(self) -> ssl.SSLContext:
        """Génère un contexte SSL sécurisé avec certifi si disponible."""
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return ssl.create_default_context()

    def _http_request(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[bytes] = None,
    ) -> Optional[bytes]:
        """
        Effectue une requête HTTP sécurisée avec gestion des timeouts et retries.
        Ne lève jamais d'exception non interceptée.
        """
        req_headers = {
            "User-Agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) FoconAssistant/1.0 ({self.name})",
            "Accept": "application/json, text/plain, */*",
        }
        if headers:
            req_headers.update(headers)

        ctx = self._get_ssl_context()
        attempt = 0

        while attempt <= self.max_retries:
            try:
                req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
                with urllib.request.urlopen(req, context=ctx, timeout=self.timeout_seconds) as resp:
                    status = getattr(resp, "status", 200)
                    if 200 <= status < 300:
                        content = resp.read()
                        self.last_fetch_time = datetime.now(timezone.utc).isoformat()
                        self.last_status = "SUCCESS"
                        return content
                    else:
                        logger.warning(f"[{self.name}] Réponse HTTP anormale {status} pour {url}")
            except urllib.error.HTTPError as he:
                logger.warning(f"[{self.name}] Erreur HTTP {he.code} ({he.reason}) sur {url}")
                if he.code in (400, 401, 403, 404):
                    # Pas de retry sur erreurs client définitives
                    break
            except urllib.error.URLError as ue:
                logger.debug(f"[{self.name}] Échec connexion (tentative {attempt+1}/{self.max_retries+1}): {ue.reason}")
            except Exception as e:
                logger.debug(f"[{self.name}] Erreur inattendue requête (tentative {attempt+1}/{self.max_retries+1}): {e}")

            attempt += 1
            if attempt <= self.max_retries:
                time.sleep(self.backoff_factor * (2 ** (attempt - 1)))

        self.last_status = "ERROR"
        return None

    def _safe_json_loads(self, content_bytes: Optional[bytes]) -> Optional[Any]:
        """Décode et valide un payload JSON sans lever d'exception."""
        if not content_bytes:
            return None
        try:
            return json.loads(content_bytes.decode("utf-8"))
        except Exception as e:
            logger.warning(f"[{self.name}] Payload JSON corrompu ou invalide: {e}")
            return None

    def normalize_indicator(
        self,
        indicator: str,
        currency_or_region: str,
        value: float,
        unit: str,
        reference_period: str,
        source_url: str,
        published_at: Optional[str] = None,
        status: str = "OFFICIAL_CURRENT",
        is_official: bool = True,
        revision: str = "INITIAL",
    ) -> Dict[str, Any]:
        """Format standard d'un indicateur macroéconomique officiel."""
        now_str = datetime.now(timezone.utc).isoformat()
        ind_clean = indicator.upper().strip()
        reg_clean = currency_or_region.upper().strip()
        per_clean = str(reference_period).strip()

        return {
            "id": f"{self.name.upper()}_{ind_clean}_{reg_clean}_{per_clean}",
            "source": self.name.upper(),
            "indicator": ind_clean,
            "currency_or_region": reg_clean,
            "value": float(value),
            "unit": unit,
            "reference_period": per_clean,
            "published_at": published_at,
            "retrieved_at": now_str,
            "status": status,
            "is_official": is_official,
            "revision": revision,
            "source_url": source_url,
        }
