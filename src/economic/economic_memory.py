# -*- coding: utf-8 -*-
"""
FATI Economic Memory (Mémoire Économique Locale SQLite)
======================================================
Stocke, indexe et interroge l'historique macroéconomique de FOCON ASSISTANT :
- Publications du calendrier économique (CPI, NFP, Taux, PIB)
- Actualités financières collectées via RSS avec scoring de sentiment
- Taux directeurs et instantanés des banques centrales
- Réactions empiriques des prix après publication d'annonces
Totalement local, autonome et sans aucun coût d'abonnement.
"""

import os
import sqlite3
import hashlib
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join("data", "economic_memory.db")


class EconomicMemory:
    """
    Gestionnaire SQLite de la mémoire économique et fondamentale.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._shared_conn = None
        if self.db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:", timeout=10.0, check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
        else:
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialise le schéma relationnel de la mémoire économique."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Événements du calendrier économique
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS economic_events (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    country TEXT,
                    currency TEXT NOT NULL,
                    impact TEXT NOT NULL, -- LOW, MEDIUM, HIGH, CRITICAL
                    event_time TEXT NOT NULL, -- ISO UTC
                    actual REAL,
                    forecast REAL,
                    previous REAL,
                    unit TEXT,
                    source TEXT DEFAULT 'PUBLIC_CALENDAR',
                    is_verified INTEGER DEFAULT 1,
                    calendar_confidence TEXT DEFAULT 'VERIFIED',
                    created_at TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON economic_events(event_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_curr ON economic_events(currency, impact)")

            # Migration douce pour les bases existantes sur disque
            try:
                cursor.execute("ALTER TABLE economic_events ADD COLUMN is_verified INTEGER DEFAULT 1")
            except Exception:
                pass
            try:
                cursor.execute("ALTER TABLE economic_events ADD COLUMN calendar_confidence TEXT DEFAULT 'VERIFIED'")
            except Exception:
                pass

            # 2. Actualités et flux RSS financiers
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS economic_news (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    link TEXT,
                    source TEXT,
                    currency TEXT,
                    sentiment_score REAL DEFAULT 0.0, -- de -1.0 (très baissier) à +1.0 (très haussier)
                    sentiment_label TEXT DEFAULT 'NEUTRAL', -- BULLISH, BEARISH, NEUTRAL
                    published_at TEXT NOT NULL,
                    summary TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_published ON economic_news(published_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_curr ON economic_news(currency)")

            # 3. Taux directeurs et snapshots macroéconomiques
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS macro_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    currency TEXT NOT NULL,
                    interest_rate REAL,
                    inflation_rate REAL,
                    gdp_growth REAL,
                    monetary_stance TEXT, -- HAWKISH, DOVISH, NEUTRAL
                    updated_at TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_macro_curr ON macro_snapshots(currency)")

            # 4. Réaction historique du marché post-annonce
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS event_reactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price_before REAL,
                    price_15m_after REAL,
                    price_1h_after REAL,
                    volatility_pips REAL,
                    direction TEXT, -- SURGE, DROP, WHIPSAW, MUTED
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()

    # ── Gestion des Événements du Calendrier ──

    def store_event(self, event: Dict[str, Any]) -> bool:
        """Enregistre ou met à jour un événement économique."""
        now_str = datetime.now(timezone.utc).isoformat()
        evt_time = str(event.get("event_time") or event.get("date_utc") or now_str)
        evt_id = str(event.get("id") or event.get("event_id") or "")
        if not evt_id:
            # Hash déterministe title + currency + time
            raw_key = f"{event.get('title')}_{event.get('currency')}_{evt_time}"
            evt_id = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

        is_verif_val = 1 if event.get("is_verified", True) else 0
        cal_conf_val = str(event.get("calendar_confidence", "VERIFIED"))

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO economic_events (
                        id, title, country, currency, impact, event_time,
                        actual, forecast, previous, unit, source, is_verified, calendar_confidence, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        actual = excluded.actual,
                        forecast = excluded.forecast,
                        previous = excluded.previous,
                        impact = excluded.impact,
                        is_verified = excluded.is_verified,
                        calendar_confidence = excluded.calendar_confidence
                """, (
                    evt_id,
                    str(event.get("title", "Unknown Event")),
                    str(event.get("country", "")),
                    str(event.get("currency", "USD")).upper(),
                    str(event.get("impact", "MEDIUM")).upper(),
                    evt_time,
                    event.get("actual"),
                    event.get("forecast"),
                    event.get("previous"),
                    str(event.get("unit", "")),
                    str(event.get("source", "PUBLIC_CALENDAR")),
                    is_verif_val,
                    cal_conf_val,
                    now_str,
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Erreur EconomicMemory.store_event: {e}")
            return False

    def store_events_bulk(self, events: List[Dict[str, Any]]) -> int:
        """Enregistre un lot d'événements."""
        stored = 0
        for ev in events:
            if self.store_event(ev):
                stored += 1
        return stored

    def get_upcoming_events(
        self,
        currency: Optional[str] = None,
        window_minutes: float = 120.0,
        min_impact: str = "HIGH",
    ) -> List[Dict[str, Any]]:
        """
        Retourne les événements à venir dans la fenêtre spécifiée (en minutes).
        Impacts acceptés : LOW, MEDIUM, HIGH, CRITICAL.
        """
        now = datetime.now(timezone.utc)
        window_end = now + timedelta(minutes=window_minutes)
        window_start = now - timedelta(minutes=15.0)  # Inclut les 15 dernières minutes

        impact_hierarchy = {
            "LOW": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            "MEDIUM": ["MEDIUM", "HIGH", "CRITICAL"],
            "HIGH": ["HIGH", "CRITICAL"],
            "CRITICAL": ["CRITICAL"],
        }
        allowed_impacts = impact_hierarchy.get(min_impact.upper(), ["HIGH", "CRITICAL"])
        impact_placeholders = ",".join(["?"] * len(allowed_impacts))

        query = f"""
            SELECT * FROM economic_events
            WHERE event_time >= ? AND event_time <= ?
            AND impact IN ({impact_placeholders})
        """
        params: List[Any] = [window_start.isoformat(), window_end.isoformat()] + allowed_impacts

        if currency:
            query += " AND currency = ?"
            params.append(currency.upper())

        query += " ORDER BY event_time ASC"

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    d = dict(row)
                    if "is_verified" in d:
                        d["is_verified"] = bool(d["is_verified"])
                    results.append(d)
                return results
        except Exception as e:
            logger.error(f"Erreur EconomicMemory.get_upcoming_events: {e}")
            return []

    # ── Gestion des Actualités RSS ──

    def store_news(self, item: Dict[str, Any]) -> bool:
        """Enregistre un article de presse avec dédoublonnage."""
        now_str = datetime.now(timezone.utc).isoformat()
        link = str(item.get("link", ""))
        title = str(item.get("title", ""))
        
        # ID unique basé sur l'URL ou le titre
        unique_key = link if link else title
        if not unique_key:
            return False
        news_id = hashlib.sha256(unique_key.encode("utf-8")).hexdigest()[:20]

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR IGNORE INTO economic_news (
                        id, title, link, source, currency, sentiment_score,
                        sentiment_label, published_at, summary, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    news_id,
                    title,
                    link,
                    str(item.get("source", "RSS")),
                    str(item.get("currency", "GENERAL")).upper(),
                    float(item.get("sentiment_score", 0.0) or 0.0),
                    str(item.get("sentiment_label", "NEUTRAL")).upper(),
                    str(item.get("published_at") or item.get("published_utc") or now_str),
                    str(item.get("summary", "")),
                    now_str,
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Erreur EconomicMemory.store_news: {e}")
            return False

    def store_news_bulk(self, news_list: List[Dict[str, Any]]) -> int:
        """Enregistre un lot d'articles de presse."""
        stored = 0
        for item in news_list:
            if self.store_news(item):
                stored += 1
        return stored

    # Alias pratiques pour la persistance
    save_economic_events = store_events_bulk
    save_news_articles = store_news_bulk


    def get_recent_news(
        self, currency: Optional[str] = None, hours: float = 24.0, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Récupère les actualités récentes pour une devise ou globales."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        query = "SELECT * FROM economic_news WHERE published_at >= ?"
        params: List[Any] = [cutoff]

        if currency and currency.upper() != "ALL":
            query += " AND (currency = ? OR currency = 'GENERAL')"
            params.append(currency.upper())

        query += " ORDER BY published_at DESC LIMIT ?"
        params.append(limit)

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return [dict(r) for r in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Erreur EconomicMemory.get_recent_news: {e}")
            return []

    # ── Gestion des Snapshots Macro ──

    def store_macro_snapshot(
        self,
        currency: str,
        interest_rate: float,
        inflation_rate: float = 0.0,
        gdp_growth: float = 0.0,
        monetary_stance: str = "NEUTRAL",
        updated_at: Optional[str] = None,
    ) -> bool:
        """Enregistre un snapshot des taux et de la posture monétaire."""
        now_str = updated_at or datetime.now(timezone.utc).isoformat()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO macro_snapshots (
                        currency, interest_rate, inflation_rate,
                        gdp_growth, monetary_stance, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    currency.upper(),
                    float(interest_rate),
                    float(inflation_rate),
                    float(gdp_growth),
                    str(monetary_stance).upper(),
                    now_str,
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Erreur EconomicMemory.store_macro_snapshot: {e}")
            return False

    def get_latest_macro_snapshot(self, currency: str) -> Optional[Dict[str, Any]]:
        """Retourne les dernières données macroéconomiques d'une devise."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM macro_snapshots
                    WHERE currency = ?
                    ORDER BY updated_at DESC LIMIT 1
                """, (currency.upper(),))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Erreur EconomicMemory.get_latest_macro_snapshot: {e}")
            return None
