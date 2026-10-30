"""
FATI Trade Journal (Journal de Bord Statistique Persistant)
==========================================================
Stocke et indexe chaque décision, exécution et clôture de trade avec ses métriques quantitatives :
- Entrée, Stop Loss, Take Profit, Volume, Coûts et Spread
- Résultat en multiples de R (R-Multiple : Net PnL / Initial Risk)
- Régime de marché, Session temporelle, Confluences techniques
- Excursion Maximale Favorable (MFE) et Défavorable (MAE)
- Base de données locale SQLite robuste avec fallback mémoire
"""

import sqlite3
import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class TradeJournal:
    """
    Journal de bord complet pour la validation statistique du système FATI.
    Persiste les décisions et les résultats dans SQLite.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            self.db_path = os.path.join(data_dir, "trade_journal.db")
        else:
            self.db_path = db_path

        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Table des trades clôturés
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        ticket INTEGER PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        side TEXT NOT NULL,
                        open_time TEXT NOT NULL,
                        close_time TEXT,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        sl_price REAL NOT NULL,
                        tp_price REAL NOT NULL,
                        volume REAL NOT NULL,
                        risk_usd REAL NOT NULL,
                        pnl_usd REAL DEFAULT 0.0,
                        result_r REAL DEFAULT 0.0,
                        spread_cost REAL DEFAULT 0.0,
                        market_regime TEXT DEFAULT 'UNKNOWN',
                        session TEXT DEFAULT 'UNKNOWN',
                        exit_reason TEXT DEFAULT 'OPEN',
                        mfe REAL DEFAULT 0.0,
                        mae REAL DEFAULT 0.0,
                        confluences TEXT DEFAULT '[]',
                        metadata TEXT DEFAULT '{}'
                    )
                """)

                # Table des décisions et rejets (NO-TRADE as an edge)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS decisions_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        action TEXT NOT NULL,
                        technical_score REAL,
                        statistical_win_prob REAL,
                        expected_value_r REAL,
                        market_regime TEXT,
                        invalidation_reason TEXT,
                        is_executed INTEGER DEFAULT 0
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"[TradeJournal] Erreur initialisation SQLite: {e}")

    def log_decision(self, decision: Dict[str, Any], is_executed: bool = False) -> int:
        """Enregistre une décision analytique (qu'elle soit acceptée ou rejetée)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now_str = datetime.utcnow().isoformat()
                cursor.execute("""
                    INSERT INTO decisions_log (
                        timestamp, symbol, timeframe, action, technical_score,
                        statistical_win_prob, expected_value_r, market_regime,
                        invalidation_reason, is_executed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    now_str,
                    str(decision.get("symbol", "UNKNOWN")),
                    str(decision.get("timeframe", "M15")),
                    str(decision.get("action", "WAIT")),
                    float(decision.get("technical_score", 0.0) or 0.0),
                    float(decision.get("statistical_win_prob", 0.0) or 0.0),
                    float(decision.get("expected_value_r", 0.0) or 0.0),
                    str(decision.get("market_regime", "UNKNOWN")),
                    str(decision.get("invalidation_reason") or decision.get("reason", "")),
                    1 if is_executed else 0,
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"[TradeJournal] Erreur log_decision: {e}")
            return -1

    def log_opened_trade(
        self,
        ticket: int,
        symbol: str,
        side: str,
        entry_price: float,
        sl: float,
        tp: float,
        volume: float = 0.1,
        risk_usd: float = 100.0,
        strategy: str = "GENERIC",
        timeframe: str = "M15",
        regime: str = "UNKNOWN",
        session: str = "UNKNOWN",
        model_version: str = "v1.0.0",
        confluences: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        lot: Optional[float] = None,
    ):
        """Alias pratique pour record_entry."""
        vol = lot if lot is not None else volume
        meta = metadata or {}
        if model_version:
            meta["model_version"] = model_version
        if timeframe:
            meta["timeframe"] = timeframe
        return self.record_entry(
            ticket=ticket,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            sl=sl,
            tp=tp,
            volume=vol,
            risk_usd=risk_usd,
            strategy=strategy,
            market_regime=regime,
            session=session,
            confluences=confluences,
            metadata=meta,
        )

    def record_entry(
        self,
        ticket: int,
        symbol: str,
        side: str,
        entry_price: float,
        sl: float,
        tp: float,
        volume: float,
        risk_usd: float,
        strategy: str = "GENERIC",
        market_regime: str = "UNKNOWN",
        session: str = "UNKNOWN",
        confluences: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Enregistre l'ouverture d'un trade réel ou simulé."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now_str = datetime.utcnow().isoformat()
                cursor.execute("""
                    INSERT OR REPLACE INTO trades (
                        ticket, symbol, strategy, side, open_time, entry_price,
                        sl_price, tp_price, volume, risk_usd, market_regime,
                        session, confluences, metadata, exit_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
                """, (
                    ticket, symbol, strategy, side.upper(), now_str,
                    entry_price, sl, tp, volume, max(risk_usd, 0.01),
                    market_regime, session,
                    json.dumps(confluences or []),
                    json.dumps(metadata or {}),
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"[TradeJournal] Erreur record_entry #{ticket}: {e}")

    def close_trade(
        self,
        ticket: int,
        exit_price: float,
        pnl_usd: float,
        exit_reason: str,
        spread_cost: float = 0.0,
        mfe: float = 0.0,
        mae: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """Alias pour record_closure."""
        return self.record_closure(
            ticket=ticket,
            exit_price=exit_price,
            pnl_usd=pnl_usd,
            exit_reason=exit_reason,
            spread_cost=spread_cost,
            mfe=mfe,
            mae=mae,
        )

    def record_closure(
        self,
        ticket: int,
        exit_price: float,
        pnl_usd: float,
        exit_reason: str,
        spread_cost: float = 0.0,
        mfe: float = 0.0,
        mae: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Enregistre la clôture d'un trade et calcule le résultat en R-Multiple.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM trades WHERE ticket = ?", (ticket,))
                row = cursor.fetchone()
                if not row:
                    # Enregistrement de secours si le trade a été ouvert avant lancement ou sans log_opened_trade
                    now_str = datetime.utcnow().isoformat()
                    result_r = round(pnl_usd / 10.0, 3)
                    cursor.execute("""
                        INSERT INTO trades (
                            ticket, symbol, strategy, side, open_time, close_time, entry_price,
                            exit_price, sl_price, tp_price, volume, risk_usd, pnl_usd, result_r,
                            spread_cost, exit_reason, mfe, mae
                        ) VALUES (?, 'UNKNOWN', 'GENERIC', 'BUY', ?, ?, ?, ?, 0.0, 0.0, 0.01, 10.0, ?, ?, ?, ?, ?, ?)
                    """, (ticket, now_str, now_str, exit_price, exit_price, pnl_usd, result_r, spread_cost, exit_reason, mfe, mae))
                    conn.commit()
                    cursor.execute("SELECT * FROM trades WHERE ticket = ?", (ticket,))
                    return dict(cursor.fetchone())

                # Protection anti-doublon : Si le trade est DÉJÀ clôturé, ne pas le ré-exécuter
                if row["exit_reason"] and str(row["exit_reason"]).upper() not in ("OPEN", "NONE", ""):
                    rec = dict(row)
                    rec["already_closed"] = True
                    return rec

                risk_usd = float(row["risk_usd"] or 1.0)
                # Calcul de l'espérance en unités R : Net PnL / Risk initial
                result_r = round(pnl_usd / max(risk_usd, 0.01), 3)
                now_str = datetime.utcnow().isoformat()

                cursor.execute("""
                    UPDATE trades SET
                        exit_price = ?,
                        pnl_usd = ?,
                        result_r = ?,
                        spread_cost = ?,
                        exit_reason = ?,
                        close_time = ?,
                        mfe = ?,
                        mae = ?
                    WHERE ticket = ?
                """, (exit_price, pnl_usd, result_r, spread_cost, exit_reason, now_str, mfe, mae, ticket))
                conn.commit()

                # Recharger l'enregistrement complet mis à jour
                cursor.execute("SELECT * FROM trades WHERE ticket = ?", (ticket,))
                updated = dict(cursor.fetchone())
                return updated
        except Exception as e:
            logger.error(f"[TradeJournal] Erreur record_closure #{ticket}: {e}")
            return None

    def get_closed_trades(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Récupère l'historique des trades clôturés."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM trades WHERE exit_reason != 'OPEN'
                    ORDER BY close_time DESC LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[TradeJournal] Erreur get_closed_trades: {e}")
            return []

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Récupère tous les trades actuellement ouverts."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM trades WHERE exit_reason = 'OPEN'")
                rows = cursor.fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[TradeJournal] Erreur get_open_trades: {e}")
            return []
