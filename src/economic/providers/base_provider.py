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

        # Traçabilité distincte Réseau/HTTP
        self.last_http_code: Optional[int] = None
        self.http_success_count: int = 0
        self.http_failure_count: int = 0

        # Traçabilité distincte Parsing / Validation Métier
        self.parsing_success_count: int = 0
        self.parsing_failure_count: int = 0
        self.business_validation_success_count: int = 0
        self.business_validation_failure_count: int = 0

        # Motifs d'échec
        self.last_error_type: Optional[str] = None  # "HTTP", "NETWORK", "EMPTY_PAYLOAD", "JSON_PARSE", "SCHEMA_VALIDATION"
        self.last_error_reason: Optional[str] = None
        self.last_error_time: Optional[str] = None
        self.last_validated_indicators_count: int = 0

    def _get_ssl_context(self) -> ssl.SSLContext:
        """Génère un contexte SSL sécurisé avec certifi si disponible."""
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return ssl.create_default_context()

    def record_error(self, error_type: str, reason: str, http_code: Optional[int] = None) -> None:
        """Enregistre un motif d'erreur détaillé et horodaté avec incrémentation des compteurs."""
        self.last_error_type = error_type
        self.last_error_reason = reason
        self.last_error_time = datetime.now(timezone.utc).isoformat()
        if http_code is not None:
            self.last_http_code = http_code
        self.last_status = "ERROR"

        if error_type in ("HTTP", "NETWORK"):
            self.http_failure_count += 1
        elif error_type in ("EMPTY_PAYLOAD", "JSON_PARSE"):
            self.parsing_failure_count += 1
        elif error_type in ("SCHEMA_VALIDATION", "BUSINESS_VALIDATION"):
            self.business_validation_failure_count += 1

    @property
    def failure_count(self) -> int:
        """Nombre total d'échecs (transport + parsing + validation métier)."""
        return self.http_failure_count + self.parsing_failure_count + self.business_validation_failure_count

    @property
    def success_count(self) -> int:
        """Nombre total de succès métier validés."""
        return self.business_validation_success_count

    def record_http_success(self, http_code: int = 200) -> None:
        """Enregistre le succès du transport HTTP."""
        self.last_http_code = http_code
        self.http_success_count += 1
        self.last_fetch_time = datetime.now(timezone.utc).isoformat()

    def record_business_success(self, count: int = 0, indicators_count: Optional[int] = None) -> None:
        """
        Enregistre la validation métier complète.
        Réinitialise les erreurs uniquement après une extraction réussie de données.
        """
        actual_count = indicators_count if indicators_count is not None else count
        self.last_validated_indicators_count = actual_count
        self.business_validation_success_count += 1
        self.parsing_success_count += 1
        self.last_status = "OK"
        self.last_error_type = None
        self.last_error_reason = None



    def get_provider_health(self) -> Dict[str, Any]:
        """
        Retourne le diagnostic d'intégrité complet séparant :
        1. Succès HTTP / Réseau
        2. Parsing et extraction des données
        3. Validation métier
        """
        return {
            "name": self.name,
            "status": self.last_status,
            "last_fetch": self.last_fetch_time,
            "last_error": self.last_error_reason,
            "last_error_type": self.last_error_type,
            "last_error_time": self.last_error_time,
            "last_http_code": self.last_http_code,
            "http_success_count": self.http_success_count,
            "http_failure_count": self.http_failure_count,
            "parsing_failure_count": self.parsing_failure_count,
            "business_validation_failure_count": self.business_validation_failure_count,
            "business_success_count": self.business_validation_success_count,
            "failure_count": self.failure_count,
            "last_validated_count": self.last_validated_indicators_count,
        }



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
        Consigne explicitement les causes réseau (timeout, DNS, SSL, refus).
        """
        req_headers = {
            "User-Agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) FoconAssistant/1.0 ({self.name})",
            "Accept": "application/json, text/plain, */*",
        }
        if headers:
            req_headers.update(headers)

        ctx = self._get_ssl_context()
        attempt = 0
        last_exception_msg = "Unknown error"

        while attempt <= self.max_retries:
            try:
                req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
                with urllib.request.urlopen(req, context=ctx, timeout=self.timeout_seconds) as resp:
                    status = getattr(resp, "status", 200)
                    self.last_http_code = status
                    if 200 <= status < 300:
                        content = resp.read()
                        if not content or len(content) == 0:
                            reason = f"Réponse HTTP {status} vide (0 octet)"
                            last_exception_msg = reason
                            if attempt == self.max_retries:
                                logger.warning(f"[{self.name}] ⚠️ {reason} sur {url}")
                                self.record_error("EMPTY_PAYLOAD", reason, http_code=status)
                            else:
                                logger.info(f"[{self.name}] 🔄 Réponse vide sur {url}, nouvel essai...")
                        else:
                            self.record_http_success(status)
                            return content
                    else:
                        reason = f"Réponse HTTP anormale {status}"
                        last_exception_msg = reason
                        if attempt == self.max_retries:
                            logger.warning(f"[{self.name}] ⚠️ {reason} pour {url}")
                            self.record_error("HTTP", reason, http_code=status)

            except urllib.error.HTTPError as he:
                reason = f"HTTP {he.code} ({he.reason})"
                logger.warning(f"[{self.name}] ⚠️ Erreur {reason} sur {url}")
                self.record_error("HTTP", reason, http_code=he.code)
                last_exception_msg = reason
                if he.code in (400, 401, 403, 404):
                    break
            except urllib.error.URLError as ue:
                reason = f"Échec réseau ({ue.reason})"
                last_exception_msg = reason
                if attempt == self.max_retries:
                    self.record_error("NETWORK", reason)
                    logger.warning(f"[{self.name}] ❌ Échec réseau définitif sur {url} après {self.max_retries+1} tentatives : {ue.reason}")
                else:
                    logger.info(f"[{self.name}] 🔄 Tentative {attempt+1}/{self.max_retries+1} échouée ({ue.reason}), nouvel essai...")
            except Exception as e:
                reason = f"Exception réseau ({type(e).__name__}: {e})"
                last_exception_msg = reason
                if attempt == self.max_retries:
                    self.record_error("NETWORK", reason)
                    logger.warning(f"[{self.name}] ❌ Erreur inattendue définitive sur {url} : {e}")
                else:
                    logger.info(f"[{self.name}] 🔄 Tentative {attempt+1}/{self.max_retries+1} échouée ({e}), nouvel essai...")


            attempt += 1
            if attempt <= self.max_retries:
                time.sleep(self.backoff_factor * (2 ** (attempt - 1)))

        if not self.last_error_reason:
            self.record_error("NETWORK", last_exception_msg)
        return None

    def _safe_json_loads(self, content_bytes: Optional[bytes]) -> Optional[Any]:
        """Décode et valide un payload JSON sans lever d'exception."""
        if not content_bytes:
            return None
        try:
            return json.loads(content_bytes.decode("utf-8"))
        except Exception as e:
            reason = f"Payload JSON corrompu ou invalide: {e}"
            logger.warning(f"[{self.name}] ⚠️ {reason}")
            self.record_error("JSON_PARSE", reason)
            return None

    @classmethod
    def evaluate_staleness(
        cls,
        indicator: str,
        reference_period: Optional[str] = None,
        effective_date: Optional[str] = None,
        published_at: Optional[str] = None,
    ) -> tuple[str, float]:
        """
        Évalue objectivement la fraîcheur économique réelle selon la fréquence propre à l'indicateur.
        Distingue une collecte HTTP récente d'une statistique économique ancienne ou en retard.
        Règles de sécurité Fail-Closed :
        - Si la période ne peut pas être interprétée : retourne ('UNKNOWN', 999.0).
        - Si la date est dans le futur (invalide pour des stats passées) : retourne ('UNKNOWN', 999.0).
        - Retourne (status, age_days) où status vaut 'OFFICIAL_CURRENT', 'STALE' ou 'UNKNOWN'.
        """
        from datetime import date
        now = datetime.now(timezone.utc)
        today = now.date()
        ind_upper = indicator.upper()

        ref_date: Optional[date] = None
        is_rate = any(k in ind_upper for k in ("RATE", "DFR", "MRO", "MLFR", "FED_FUNDS"))

        # 1. Pour les taux directeurs de banque centrale, priorité absolue à la date d'effet officielle
        if is_rate and effective_date:
            try:
                ref_date = date.fromisoformat(str(effective_date)[:10])
            except Exception:
                pass


        # 2. Pour les statistiques économiques (CPI, chômage, PIB, HICP), priorité absolue à la période de référence
        # Ne jamais remplacer la période macroéconomique par un horodatage de publication ou de collecte !
        if not ref_date and reference_period:
            per = str(reference_period).strip().upper()
            try:
                # Format trimestriel YYYYQ1, YYYY-Q1, YYYY-Q2 (doit être testé AVANT les longueurs 7)
                if "Q" in per:
                    parts = per.replace("-", "").split("Q")
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        y = int(parts[0])
                        q = int(parts[1])
                        if 1 <= q <= 4:
                            q_end_month = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
                            m, day = q_end_month[q]
                            ref_date = date(y, m, day)
                # Format mensuel BLS YYYYM09 ou YYYY-M09
                elif "M" in per:
                    parts = per.replace("-", "").split("M")
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        y = int(parts[0])
                        m = int(parts[1])
                        if 1 <= m <= 12:
                            day = 28 if m == 2 else (30 if m in (4, 6, 9, 11) else 31)
                            ref_date = date(y, m, day)
                # Format ISO complet YYYY-MM-DD
                elif len(per) == 10 and per[:4].isdigit() and per[4] == "-" and per[5:7].isdigit() and per[7] == "-" and per[8:10].isdigit():
                    ref_date = date.fromisoformat(per)
                # Format mensuel ISO YYYY-MM (ex: 2025-12, 2026-08)
                elif len(per) == 7 and per[:4].isdigit() and per[4] == "-" and per[5:7].isdigit():
                    y = int(per[:4])
                    m = int(per[5:7])
                    if 1 <= m <= 12:
                        day = 28 if m == 2 else (30 if m in (4, 6, 9, 11) else 31)
                        ref_date = date(y, m, day)
                # Format annuel YYYY
                elif len(per) == 4 and per.isdigit():
                    ref_date = date(int(per), 12, 31)
            except Exception:
                ref_date = None

        # 3. Repli subsidiaire sur effective_date pour tout autre indicateur
        if not ref_date and effective_date:
            try:
                ref_date = date.fromisoformat(str(effective_date)[:10])
            except Exception:
                pass

        # 4. Repli de dernier recours sur published_at UNIQUEMENT si aucune période de référence n'a pu être résolue
        if not ref_date and published_at:
            try:
                ref_date = date.fromisoformat(str(published_at)[:10])
            except Exception:
                pass


        # Règle Fail-Closed fondamentale : Si impossible d'interpréter la période,
        # ne JAMAIS classer comme actuelle avec un âge nul. Retourner UNKNOWN avec âge pénalisant.
        if not ref_date:
            return "UNKNOWN", 999.0

        is_rate = any(k in ind_upper for k in ("RATE", "DFR", "MRO", "MLFR", "FED_FUNDS"))

        # Vérification des dates futures
        if ref_date > today:
            # Pour un taux directeur, une date d'effet dans les 7 prochains jours correspond à
            # une annonce de politique monétaire imminente validée par la banque centrale
            if is_rate and 0 < (ref_date - today).days <= 7:
                return "OFFICIAL_CURRENT", 0.0
            # Pour une statistique passée (CPI, PIB, chômage) ou date aberrante : rejet strict
            return "UNKNOWN", 999.0

        age_days = max(0.0, float((today - ref_date).days))

        # Cadences de publication économiques standards :
        # - Taux directeurs : réévaluation toutes les 6-8 semaines (stale si > 90 jours sans décision)
        # - Statistiques mensuelles (CPI, Chômage, NFP, HICP) : délai usuel 2-5 semaines (stale si > 75 jours)
        # - Statistiques trimestrielles (PIB) : délai usuel 6-8 semaines (stale si > 150 jours)
        # - Statistiques annuelles : publication annuelle (stale si > 370 jours)
        is_annual = bool(reference_period and ((len(str(reference_period).strip()) == 4 and str(reference_period).strip().isdigit()) or ("ANNUAL" in str(reference_period).upper())))
        if is_rate:
            max_allowed = 90.0
        elif any(k in ind_upper for k in ("GDP", "PIB", "GROWTH")):
            max_allowed = 150.0
        elif is_annual:
            max_allowed = 370.0
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
        Évalue automatiquement le statut (OFFICIAL_CURRENT, STALE, UNKNOWN) selon l'âge et la validité.
        """
        import math
        now_str = datetime.now(timezone.utc).isoformat()
        ind_clean = indicator.upper().strip()
        reg_clean = currency_or_region.upper().strip()
        per_clean = str(reference_period).strip()

        # Validation numérique de la valeur
        try:
            val_float = float(value)
            if math.isnan(val_float) or math.isinf(val_float):
                val_float = 0.0
                is_val_valid = False
            else:
                is_val_valid = True
        except (ValueError, TypeError):
            val_float = 0.0
            is_val_valid = False

        # Évaluation dynamique de la fraîcheur si le statut n'est pas forcé (ex: HISTORICAL)
        calc_status, age_days = self.evaluate_staleness(
            indicator=ind_clean,
            reference_period=per_clean,
            effective_date=effective_date,
            published_at=published_at,
        )

        if not is_val_valid:
            final_status = "UNKNOWN"
            age_days = 999.0
        else:
            final_status = status or calc_status

        return {
            "id": f"{self.name.upper()}_{ind_clean}_{reg_clean}_{per_clean}",
            "source": self.name.upper(),
            "indicator": ind_clean,
            "currency_or_region": reg_clean,
            "value": val_float,
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
