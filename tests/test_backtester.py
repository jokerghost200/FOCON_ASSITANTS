"""
Tests unitaires pour BacktestEngine.
"""

import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

from src.trading.backtester import BacktestEngine
from src.trading.risk_manager import RiskManager


class TestBacktestEngine(unittest.TestCase):
    def setUp(self):
        self.mock_mt5 = MagicMock()
        self.mock_mt5.connected = True
        self.mock_mt5.connect.return_value = True

        # Génération de 120 bougies simulées
        times = [1700000000 + i * 900 for i in range(120)]
        prices = 1.1000 + np.cumsum(np.random.normal(0.00005, 0.0002, 120))
        rates = []
        for i, t in enumerate(times):
            p = prices[i]
            rates.append({
                "time": t,
                "open": p,
                "high": p + 0.0005,
                "low": p - 0.0005,
                "close": p + 0.0001,
                "tick_volume": 100,
                "spread": 10,
            })
        self.mock_mt5.get_rates.return_value = rates
        self.mock_mt5.get_symbol_info.return_value = {
            "name": "EURUSD",
            "point": 0.00001,
            "trade_tick_size": 0.00001,
            "trade_tick_value": 1.0,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
            "spread": 10,
        }
        self.engine = BacktestEngine(self.mock_mt5)

    def test_backtest_runs_and_produces_metrics(self):
        res = self.engine.run_backtest(
            symbol="EURUSD",
            timeframe="M15",
            num_bars=120,
            initial_capital=100.0,
            risk_profile="BALANCED",
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["initial_capital"], 100.0)
        self.assertIn("conclusion", res)
        self.assertGreater(len(res["conclusion"]), 100)
        self.assertIn("equity_curve", res)
        self.assertIsInstance(res["trades"], list)

    def test_backtest_micro_capital_tier_detected(self):
        res = self.engine.run_backtest(
            symbol="EURUSD",
            timeframe="M15",
            num_bars=120,
            initial_capital=50.0,
            risk_profile="BALANCED",
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["tier_info"]["tier_id"], "MICRO")
        self.assertIn("Micro", res["conclusion"])

    def test_backtest_zero_trades_diagnostic(self):
        conclusion = self.engine._generate_ai_conclusion(
            symbol="BTCUSD",
            timeframe="M15",
            num_bars=300,
            initial_capital=50.0,
            final_balance=50.0,
            net_profit_usd=0.0,
            net_return_pct=0.0,
            tier_info={"tier_id": "MICRO", "name": "Micro-Capital"},
            total_trades=0,
            win_rate=0.0,
            profit_factor=1.0,
            max_drawdown_pct=0.0,
            exits_tp=0,
            exits_sl=0,
            exits_be=0,
            blocked_by_micro_protection=12,
            blocked_by_spread=0,
            blocked_by_confidence=0,
            total_signals_generated=12,
        )
        self.assertIn("DIAGNOSTIC DU 0 TRADE", conclusion)
        self.assertIn("PROTECTION ACTIVE", conclusion)
        self.assertIn("FOREX", conclusion)
        self.assertIn("BTCUSD", conclusion)

    def test_backtest_with_fixed_lot_size(self):
        """Vérifie que le mode lot fixe est bien respecté dans le backtest."""
        res = self.engine.run_backtest(
            symbol="EURUSD",
            timeframe="M15",
            num_bars=120,
            initial_capital=500.0,
            fixed_lot_size=0.20,
            strategy_filter="ALL",
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["fixed_lot_size"], 0.20)
        self.assertIn("Lot Fixe (0.20)", res["conclusion"])
        for trade in res.get("trades", []):
            self.assertEqual(trade["lot"], 0.20)

    def test_backtest_strategy_filter_modes(self):
        """Vérifie que les modes de filtrage de stratégie ('CRT', 'ORDER_BLOCKS', 'BOOMERANG') s'exécutent sans erreur."""
        for strat in ("CRT", "ORDER_BLOCKS", "BOOMERANG"):
            res = self.engine.run_backtest(
                symbol="EURUSD",
                timeframe="M15",
                num_bars=120,
                initial_capital=200.0,
                strategy_filter=strat,
            )
            self.assertTrue(res["success"])
            self.assertEqual(res["strategy_filter"], strat)
            self.assertIn(strat, res["conclusion"])


if __name__ == "__main__":
    unittest.main()
