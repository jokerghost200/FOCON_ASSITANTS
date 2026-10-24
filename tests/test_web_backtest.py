import unittest
from unittest.mock import MagicMock, patch
from src.web.app import create_app
from src.web.state import WebState


class TestWebBacktest(unittest.TestCase):
    def setUp(self):
        self.fake_orch = MagicMock()
        self.fake_orch.risk_manager = MagicMock()
        self.fake_orch.mt5 = MagicMock()
        self.fake_orch.mt5.connected = True
        self.fake_exec = MagicMock()
        self.fake_chat = MagicMock()
        self.fake_tm = MagicMock()
        self.config = {
            "risk_management": {
                "default_risk_percent": 1.0,
                "max_daily_loss_percent": 5.0,
                "max_lot_size": 1.0,
                "max_open_positions": 3,
                "fixed_lot_size": 0.0,
                "risk_profile": "BALANCED",
            },
            "trading": {
                "automation_mode": "assist",
                "allowed_market_types": ["Forex"],
                "timeframes": ["M15"],
            },
        }
        self.state = WebState(
            self.fake_orch, self.fake_exec, self.fake_chat, self.config, self.fake_tm
        )
        self.app = create_app(self.state)
        self.client = self.app.test_client()

    def test_backtest_status_initial(self):
        res = self.client.get("/api/backtest/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["ok"])
        self.assertFalse(data["running"])
        self.assertEqual(data["progress"], 0.0)

    @patch("src.trading.backtester.BacktestEngine.run_backtest")
    def test_backtest_run_triggers_worker(self, mock_run):
        mock_run.return_value = {
            "stats": {
                "total_trades": 5,
                "win_rate": 80.0,
                "profit_factor": 2.5,
                "net_profit_usd": 15.5,
                "net_profit_pct": 15.5,
                "final_balance": 115.5,
                "max_drawdown_pct": 2.0,
                "max_drawdown_usd": 2.5,
                "break_even_trades": 1,
                "pending_orders_placed": 3,
                "pending_orders_filled": 2,
            },
            "trades": [],
            "report_text": "Simulation Test OK",
        }

        res = self.client.post(
            "/api/backtest/run",
            json={
                "symbol": "EURUSD",
                "timeframe": "M15",
                "bars": 300,
                "capital": 100.0,
                "strategy": "CRT",
                "profile": "BALANCED",
                "fixed_lot": 0.01,
                "direction": "BOTH",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["ok"])

        # Wait briefly for worker thread to complete
        if self.state.backtest_thread:
            self.state.backtest_thread.join(timeout=3.0)

        status_res = self.client.get("/api/backtest/status")
        status_data = status_res.get_json()
        self.assertFalse(status_data["running"])
        self.assertIsNotNone(status_data["results"])
        self.assertEqual(status_data["results"]["stats"]["total_trades"], 5)

    def test_save_config_with_fixed_lot_and_profile(self):
        with patch.object(self.state, "save_config"):
            res = self.client.post(
                "/api/config",
                json={
                    "risk_management": {
                        "default_risk_percent": 1.5,
                        "fixed_lot_size": 0.02,
                        "risk_profile": "AGGRESSIVE",
                    }
                },
            )
            self.assertEqual(res.status_code, 200)
            self.assertEqual(self.config["risk_management"]["fixed_lot_size"], 0.02)
            self.assertEqual(self.config["risk_management"]["risk_profile"], "AGGRESSIVE")
            self.assertEqual(self.fake_orch.risk_manager.fixed_lot_size, 0.02)


if __name__ == "__main__":
    unittest.main()
