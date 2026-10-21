import sqlite3
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryEngine:
    """
    Phase 3: Memory & Performance Engine.
    Handles storing trade metadata to analyze the strategy's expectancy over time.
    """
    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Creates the SQLite database schema if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        ticket INTEGER PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        session TEXT,
                        amd_phase TEXT,
                        has_fvg INTEGER,
                        ob_type TEXT,
                        entry_price REAL,
                        sl_price REAL,
                        tp_price REAL,
                        initial_rr REAL,
                        risk_usd REAL,
                        status TEXT DEFAULT 'OPEN',
                        profit_usd REAL DEFAULT 0.0,
                        profit_r REAL DEFAULT 0.0,
                        open_time TIMESTAMP,
                        close_time TIMESTAMP
                    )
                ''')
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize Memory Engine DB: {e}")

    def log_trade_open(self, ticket: int, symbol: str, direction: str, strategy: str, 
                       context: Dict[str, Any], entry: float, sl: float, tp: float, risk_usd: float):
        """
        Logs a new trade being opened with its full context (Phase 3 requirements).
        """
        try:
            sl_dist = abs(entry - sl)
            tp_dist = abs(tp - entry)
            initial_rr = tp_dist / sl_dist if sl_dist > 0 else 0.0

            session = context.get('session', 'Unknown')
            amd_phase = context.get('amd_phase', 'Unknown')
            has_fvg = 1 if context.get('has_fvg', False) else 0
            ob_type = context.get('ob_type', 'Unknown')

            now = datetime.utcnow().isoformat()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO trades 
                    (ticket, symbol, direction, strategy_name, session, amd_phase, has_fvg, ob_type, 
                     entry_price, sl_price, tp_price, initial_rr, risk_usd, status, open_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
                ''', (ticket, symbol, direction, strategy, session, amd_phase, has_fvg, ob_type,
                      entry, sl, tp, initial_rr, risk_usd, now))
                conn.commit()
                logger.info(f"[MemoryEngine] Trade {ticket} logged in DB (Strategy: {strategy}).")
        except Exception as e:
            logger.error(f"[MemoryEngine] Error logging trade {ticket}: {e}")

    def log_trade_close(self, ticket: int, profit_usd: float):
        """
        Updates a trade when it is closed, calculating its actual profit in R multiples.
        """
        try:
            now = datetime.utcnow().isoformat()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Fetch risk_usd to calculate profit_r
                cursor.execute('SELECT risk_usd FROM trades WHERE ticket = ?', (ticket,))
                row = cursor.fetchone()
                profit_r = 0.0
                if row and row[0] and row[0] > 0:
                    profit_r = profit_usd / row[0]

                cursor.execute('''
                    UPDATE trades
                    SET status = 'CLOSED', profit_usd = ?, profit_r = ?, close_time = ?
                    WHERE ticket = ?
                ''', (profit_usd, profit_r, now, ticket))
                conn.commit()
                logger.info(f"[MemoryEngine] Trade {ticket} closed. Profit: {profit_r:.2f}R")
        except Exception as e:
            logger.error(f"[MemoryEngine] Error closing trade {ticket}: {e}")

    def calculate_expectancy(self, symbol: str, session: str, strategy: str) -> Optional[float]:
        """
        Calculates the expectancy in R for a given setup.
        Expectancy = (WinRate * AvgWinR) + (LossRate * AvgLossR)
        Note: The standard formula adds them if AvgLossR is negative, 
        or subtracts them if you treat AvgLossR as a positive absolute value.
        Here we assume profit_r is positive for wins and negative for losses.
        Returns None if not enough data (e.g. less than 5 trades).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT profit_r FROM trades 
                    WHERE symbol = ? AND session = ? AND strategy_name = ? AND status = 'CLOSED'
                ''', (symbol, session, strategy))
                
                rows = cursor.fetchall()
                if len(rows) < 3: # Need at least a few trades to evaluate
                    return None
                    
                profits = [r[0] for r in rows]
                wins = [p for p in profits if p > 0]
                losses = [p for p in profits if p <= 0]
                
                win_rate = len(wins) / len(profits)
                loss_rate = len(losses) / len(profits)
                
                avg_win = sum(wins) / len(wins) if wins else 0.0
                avg_loss = sum(losses) / len(losses) if losses else 0.0
                
                # Since losses are already negative numbers:
                expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
                return expectancy
        except Exception as e:
            logger.error(f"[MemoryEngine] Error calculating expectancy: {e}")
            return None

    def evaluate_strategy_validity(self, symbol: str, session: str, strategy: str) -> Dict[str, Any]:
        """
        Checks if a strategy should be disabled for a specific context.
        """
        exp = self.calculate_expectancy(symbol, session, strategy)
        if exp is not None and exp < -0.1:
            return {
                "valid": False,
                "reason": f"Attention, sur {symbol} en session {session}, la stratégie {strategy} a une expectancy négative de {exp:.2f}R, on désactive."
            }
        return {
            "valid": True, 
            "reason": f"Expectancy OK ({exp:.2f}R)" if exp is not None else "Not enough data"
        }

    def sync_open_trades_with_mt5(self, mt5_connector):
        """
        Checks currently open trades in the DB against MT5 history.
        If a trade is no longer open in MT5, update its status in DB.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ticket FROM trades WHERE status = 'OPEN'")
                open_tickets = [row[0] for row in cursor.fetchall()]
                
            for ticket in open_tickets:
                # Use MT5 connector to get deal info
                deal = mt5_connector.get_deal_info_from_history(ticket)
                if deal:
                    # Trade is closed
                    profit = deal.get('profit', 0.0)
                    self.log_trade_close(ticket, profit)
                    
        except Exception as e:
            logger.error(f"[MemoryEngine] Error syncing with MT5: {e}")
