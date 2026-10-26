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

    def evaluate_staleness(
        self,
        indicator: str,
        reference_period: str,
        effective_date: Optional[str] = None,
        published_at: Optional[str] = None,
    ) -> tuple[str, float]:
        """
        Évalue objectivement la fraîcheur économique réelle selon la fréquence propre à l'indicateur.
        Distingue une collecte HTTP récente d'une statistique économique ancienne ou en retard.
        Retourne (status, age_days) où status vaut 'OFFICIAL_CURRENT' ou 'STALE'.
        """
        from datetime import date
        now = datetime.now(timezone.utc)
        today = now.date()
        ind_upper = indicator.upper()

        ref_date: Optional[date] = None

        # 1. Priorité à la date d'effet explicite (ex: taux directeurs de banque centrale)
        if effective_date:
            try:
                ref_date = date.fromisoformat(str(effective_date)[:10])
            except Exception:
                pass

        # 2. Date de publication officielle si disponible
        if not ref_date and published_at:
            try:
                ref_date = date.fromisoformat(str(published_at)[:10])
            except Exception:
                pass

        # 3. Parsing de la période de référence
        if not ref_date and reference_period:
            per = str(reference_period).strip().upper()
            try:
                # Format ISO YYYY-MM-DD
                if len(per) == 10 and per[4] == "-" and per[7] == "-":
                    ref_date = date.fromisoformat(per)
                # Format mensuel BLS YYYYM09 ou YYYY-M09
                elif "M" in per:
                    parts = per.replace("-", "").split("M")
                    y = int(parts[0])
                    m = int(parts[1])
                    # Dernier jour du mois approximé (28 pour fév, 30/31 pour les autres)
                    day = 28 if m == 2 else (30 if m in (4, 6, 9, 11) else 31)
                    ref_date = date(y, m, day)
                # Format mensuel ISO YYYY-MM
                elif len(per) == 7 and per[4] == "-":
                    y = int(per[:4])
                    m = int(per[5:7])
                    day = 28 if m == 2 else (30 if m in (4, 6, 9, 11) else 31)
                    ref_date = date(y, m, day)
                # Format trimestriel YYYYQ1 ou YYYY-Q1
                elif "Q" in per:
                    parts = per.replace("-", "").split("Q")
                    y = int(parts[0])
                    q = int(parts[1])
                    q_end_month = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
                    m, day = q_end_month.get(q, (12, 31))
                    ref_date = date(y, m, day)
                # Format annuel YYYY
                elif len(per) == 4 and per.isdigit():
                    ref_date = date(int(per), 12, 31)
            except Exception:
                ref_date = None

        if not ref_date:
            # Période indéterminée : conservation du statut sans garantie
            return "OFFICIAL_CURRENT", 0.0

        age_days = max(0.0, float((today - ref_date).days))

        # Cadences de publication économiques standards :
        # - Taux directeurs : réévaluation toutes les 6-8 semaines (stale si > 90 jours sans décision)
        # - Statistiques mensuelles (CPI, Chômage, NFP, HICP) : délai usuel 2-5 semaines (stale si > 75 jours)
        # - Statistiques trimestrielles (PIB) : délai usuel 6-8 semaines (stale si > 150 jours)
        if any(k in ind_upper for k in ("RATE", "DFR", "MRO", "MLFR", "FED_FUNDS")):
            max_allowed = 90.0
        elif any(k in ind_upper for k in ("GDP", "PIB", "GROWTH")):
            max_allowed = 150.0
        else:
            max_allowed = 75.0

        status = "OFFICIAL_CURRENT" if age_days <= max_allowed else "STALE"
        return status, age_days

    def normalize_indicator(
        self,
        indicator: str,
        currency_or_region: str,
        value: float,
        unit: str,
        reference_period: str,
        source_url: str,
        published_at: Optional[str] = None,
        effective_date: Optional[str] = None,
        status: Optional[str] = None,
        is_official: bool = True,
        revision: str = "INITIAL",
    ) -> Dict[str, Any]:
        """
        Format standard d'un indicateur macroéconomique officiel.
        Évalue automatiquement le statut (OFFICIAL_CURRENT vs STALE) selon l'âge réel de la donnée.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        ind_clean = indicator.upper().strip()
        reg_clean = currency_or_region.upper().strip()
        per_clean = str(reference_period).strip()

        # Évaluation dynamique de la fraîcheur si le statut n'est pas forcé (ex: HISTORICAL)
        calc_status, age_days = self.evaluate_staleness(
            indicator=ind_clean,
            reference_period=per_clean,
            effective_date=effective_date,
            published_at=published_at,
        )
        final_status = status or calc_status

        return {
            "id": f"{self.name.upper()}_{ind_clean}_{reg_clean}_{per_clean}",
            "source": self.name.upper(),
            "indicator": ind_clean,
            "currency_or_region": reg_clean,
            "value": float(value),
            "unit": unit,
            "reference_period": per_clean,
            "effective_date": effective_date,
            "published_at": published_at,
            "retrieved_at": now_str,
            "age_days": age_days,
            "status": final_status,
            "is_official": is_official,
            "revision": revision,
            "source_url": source_url,
        }
