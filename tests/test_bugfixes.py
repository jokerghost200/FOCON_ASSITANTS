import unittest
import tempfile
import os
import pandas as pd
from unittest.mock import MagicMock
from src.core.boomerang_detector import BoomerangDetector
from src.learning.trade_memory import TradeMemoryEngine
from src.learning.adaptive_weights import AdaptiveWeightsEngine, DEFAULT_WEIGHTS
from src.learning.market_adapt_sentinel import MarketAdaptationSentinel
from src.core.technical_indicators import TechnicalIndicators


class TestBugfixes(unittest.TestCase):

    def test_boomerang_detector_bos_fvg_truthiness(self):
        detector = BoomerangDetector()
        
        # Création d'un dataframe synthétique suffisant (> 50 bougies)
        data = {
            'open': [1.1000 + i * 0.0001 for i in range(60)],
            'high': [1.1005 + i * 0.0001 for i in range(60)],
            'low': [1.0995 + i * 0.0001 for i in range(60)],
            'close': [1.1002 + i * 0.0001 for i in range(60)],
            'volume': [100.0] * 60,
        }
        df = pd.DataFrame(data)
        ti = TechnicalIndicators()
        df = ti.add_all_indicators(df)
        
        # 1. amd_context avec chaînes 'None' : ne doit PAS activer les bonus BOS ou FVG
        amd_none = {"Phase": "Distribution", "BOS_MSS": "None", "FVG": "None", "Liquidity_Grab": "None"}
        res_none = detector.detect(df, "EURUSD", "M5", amd_context=amd_none)
        self.assertFalse(any("BOS/MSS confirmed" in r for r in res_none.reasons))
        self.assertFalse(any("FVG present" in r for r in res_none.reasons))

        # 2. amd_context avec Bearish BOS alors que la direction est BUY : ne doit PAS activer le bonus
        amd_bearish = {"Phase": "Distribution", "BOS_MSS": "Bearish MSS", "FVG": "Bearish FVG", "Liquidity_Grab": "None"}
        res_bearish = detector.detect(df, "EURUSD", "M5", amd_context=amd_bearish)
        # L'impulsion sur ce df haussier sera BUY, donc Bearish MSS ne doit pas être validé pour BUY
        if res_bearish.direction == 'BUY':
            self.assertFalse(any("BOS/MSS confirmed" in r for r in res_bearish.reasons))

    def test_trade_memory_check_expectancy_strategy_names(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fpath = os.path.join(tmp_dir, "trade_memory.json")
            tm = TradeMemoryEngine(file_path=str(fpath))
            
            # Enregistrement d'un trade avec strategy OB_FVG_RETEST
            tm.record_entry(
                ticket=101,
                symbol="EURUSD",
                action="BUY",
                lot=0.1,
                price=1.1000,
                sl=1.0950,
                tp=1.1100,
                snapshot={
                    "session": "LONDON",
                    "has_ob": True,
                    "fvg": "Bullish FVG",
                    "sweep": "None",
                    "initial_rr": 2.0,
                    "risk_usd": 50.0,
                }
            )
            tm.record_exit(ticket=101, close_price=1.1100, final_profit_usd=100.0, exit_reason="TP_HIT")
            
            # Enregistrement d'un second et troisième trade pour atteindre le minimum requis de 3
            for tkt in [102, 103]:
                tm.record_entry(
                    ticket=tkt,
                    symbol="EURUSD",
                    action="BUY",
                    lot=0.1,
                    price=1.1000,
                    sl=1.0950,
                    tp=1.1100,
                    snapshot={
                        "session": "LONDON",
                        "has_ob": True,
                        "fvg": "Bullish FVG",
                        "sweep": "None",
                        "initial_rr": 2.0,
                        "risk_usd": 50.0,
                    }
                )
                tm.record_exit(ticket=tkt, close_price=1.1100, final_profit_usd=100.0, exit_reason="TP_HIT")

            res = tm.check_expectancy("EURUSD", "LONDON", "OB_FVG_RETEST")
            self.assertIn(res["status"], ("ACTIVE", "NEUTRAL"))
            self.assertEqual(res["trades_count"], 3)

    def test_market_adapt_sentinel_with_trade_memory_dict(self):
        orch = MagicMock()
        tm_mock = MagicMock()
        # Simuler des trades clôturés avec pertes consécutives via final_profit_usd
        tm_mock.get_closed_trades.return_value = [
            {"ticket": 2, "status": "CLOSED", "final_profit_usd": -15.0, "close_time": 200},
            {"ticket": 1, "status": "CLOSED", "final_profit_usd": -12.0, "close_time": 100},
        ]
        orch.trade_memory = tm_mock
        orch.mt5.connected = True
        orch.mt5.get_account_info.return_value = {"balance": 500.0, "equity": 500.0}

        config = {
            "risk_management": {"default_risk_percent": 1.0, "min_confidence": 65.0, "max_open_positions": 3},
            "trading": {"allowed_market_types": ["Forex"]}
        }

        result = MarketAdaptationSentinel.evaluate_market_and_account(orch, config)
        self.assertEqual(result["consecutive_losses"], 2)
        self.assertTrue(result["need_adaptation"])

    def test_adaptive_weights_all_weight_keys(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fpath = os.path.join(tmp_dir, "adaptive_weights.json")
            engine = AdaptiveWeightsEngine(file_path=str(fpath), learning_rate=0.5)

            sample_trades = [
                {
                    "final_profit_usd": 20.0,
                    "snapshot": {
                        "has_ob": True,
                        "has_boomerang": True,
                        "htf_trend": "BULLISH",
                        "ema_dist_atr": 1.2,
                        "rsi_14": 50.0,
                        "amd_phase": "Expansion",
                        "fvg": "Bullish FVG",
                    }
                },
                {
                    "final_profit_usd": 30.0,
                    "snapshot": {
                        "has_ob": True,
                        "has_boomerang": True,
                        "htf_trend": "BULLISH",
                        "ema_dist_atr": 1.5,
                        "rsi_14": 48.0,
                        "amd_phase": "Distribution",
                        "fvg": "Bullish FVG",
                    }
                },
                {
                    "final_profit_usd": 25.0,
                    "snapshot": {
                        "has_ob": True,
                        "has_boomerang": True,
                        "htf_trend": "BULLISH",
                        "ema_dist_atr": 0.8,
                        "rsi_14": 52.0,
                        "amd_phase": "Distribution",
                        "fvg": "Bullish FVG",
                    }
                }
            ]

            res = engine.recalibrate(sample_trades)
            self.assertEqual(res["status"], "RECALIBRATED")
            # Vérifier que trend_ema_weight et amd_expansion_weight ont bien été ajustés
            self.assertIn("trend_ema_weight", res["adjustments"])
            self.assertIn("amd_expansion_weight", res["adjustments"])


    def test_symbol_cooldown_blocks_reentry(self):
        from unittest.mock import patch
        from src.ai.orchestrator import AIOrchestrator
        orch = AIOrchestrator.__new__(AIOrchestrator)
        orch._symbol_cooldowns = {}

        orch.set_symbol_cooldown("GBPJPY", duration_seconds=600, reason="Perte SL", direction="BUY")
        
        in_cd, reason = orch.is_symbol_in_cooldown("GBPJPY", "BUY")
        self.assertTrue(in_cd)
        self.assertIn("Perte SL", reason)

        # Different symbol is not in cooldown
        in_cd_other, _ = orch.is_symbol_in_cooldown("EURUSD", "BUY")
        self.assertFalse(in_cd_other)

        # Different direction (SELL) is not in cooldown
        in_cd_sell, _ = orch.is_symbol_in_cooldown("GBPJPY", "SELL")
        self.assertFalse(in_cd_sell)

    def test_pending_order_manager_sync_and_clear(self):
        from src.trading.pending_order_manager import PendingOrderManager, PendingOBOrder

        mt5_mock = MagicMock()
        mt5_mock.connected = True
        # MT5 has no pending orders left (all closed/cancelled)
        mt5_mock.get_pending_orders.return_value = []
        mt5_mock.get_open_positions.return_value = []

        mgr = PendingOrderManager(
            mt5_connector=mt5_mock,
            ob_detector=MagicMock(),
            tech_indicators=MagicMock(),
            market_data=MagicMock(),
            risk_manager=MagicMock(),
        )

        # Add a mock active order
        mgr._pending_orders[585364059] = PendingOBOrder(
            ticket=585364059,
            symbol="AUDNZD",
            direction="SELL",
            entry_price=1.23303,
            sl=1.23500,
            tp=1.23000,
            volume=0.01,
            ob_high=1.23350,
            ob_low=1.23250,
            ob_score=70.0,
            timeframe="M15",
            status="ACTIVE"
        )

        # sync_with_mt5 should detect it is no longer in MT5 and mark it CANCELLED
        mgr.sync_with_mt5()
        self.assertEqual(mgr._pending_orders[585364059].status, "CANCELLED")

        # clear_inactive should purge it
        cleaned = mgr.clear_inactive()
        self.assertEqual(cleaned, 1)
        self.assertNotIn(585364059, mgr._pending_orders)

    def test_safe_margin_capping_prevents_margin_exhaustion(self):
        from unittest.mock import patch
        from src.trading.risk_manager import RiskManager

        rm = RiskManager(fixed_lot_size=1.0)
        # Sur un compte de 50 EUR avec 45 EUR de marge libre
        account_info = {"balance": 50.0, "margin_free": 45.0}
        symbol_info = {
            "name": "GBPJPY",
            "digits": 3,
            "point": 0.001,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
            "trade_contract_size": 100000.0,
            "trade_tick_size": 0.001,
            "trade_tick_value": 0.65,
            "spread": 10,
        }
        proposal = {
            "action": "BUY",
            "entry": 211.0,
            "sl": 210.5,
            "tp": 212.5,
            "confidence": 85.0,
        }

        # order_calc_margin mock for MT5
        with patch("MetaTrader5.order_calc_margin", return_value=235.0): # 1 lot requires ~235 EUR
            val = rm.validate_proposal(proposal, account_info, symbol_info)
            self.assertTrue(val["valid"])
            # Sur un compte < 100 EUR, le lot doit être sécurisé (max 0.02 ou 25% de marge libre)
            self.assertLessEqual(val["proposal"]["lot"], 0.05)
            self.assertLess(val["proposal"]["lot"], 0.20)  # Ne doit JAMAIS allouer 0.20+ lots !


if __name__ == "__main__":
    unittest.main()

