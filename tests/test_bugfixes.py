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

    def test_order_execution_invalid_expiration_retries_with_gtc(self):
        from unittest.mock import patch, MagicMock
        from src.trading.trade_execution import TradeExecutionEngine
        import MetaTrader5 as mt5

        mock_mt5 = MagicMock()
        mock_mt5.get_account_info.return_value = {"balance": 20000.0, "equity": 20000.0}
        mock_mt5.get_open_positions.return_value = []
        mock_mt5.get_daily_profit.return_value = 0.0
        mock_mt5.get_symbol_info.return_value = {"trade_tick_value": 1.0, "trade_tick_size": 0.01}
        te = TradeExecutionEngine(mt5_connector=mock_mt5, mode="assist")

        # Mock order_check failure with code 10022 then success
        err_res = MagicMock()
        err_res.retcode = 10022
        err_res.comment = "Invalid expiration"

        ok_res = MagicMock()
        ok_res.retcode = mt5.TRADE_RETCODE_DONE
        ok_res.margin = 10.0
        ok_res.margin_free = 90.0

        with patch("MetaTrader5.order_check", side_effect=[err_res, ok_res]):
            req = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": "LTCUSD",
                "volume": 1.0,
                "type": mt5.ORDER_TYPE_SELL_LIMIT,
                "price": 57.49,
                "sl": 59.11,
                "tp": 54.25,
                "type_time": mt5.ORDER_TIME_SPECIFIED,
                "expiration": 1234567,
            }
            res = te.check_order(req)
            self.assertTrue(res["valid"])
            self.assertEqual(req["type_time"], mt5.ORDER_TIME_GTC)
            self.assertNotIn("expiration", req)

        # Mock order_send failure with code 10022 then success
        err_send = MagicMock()
        err_send.retcode = 10022
        err_send.comment = "Invalid expiration"

        ok_send = MagicMock()
        ok_send.retcode = mt5.TRADE_RETCODE_DONE
        ok_send.deal = 98765
        ok_send.order = 98765
        ok_send.volume = 1.0
        ok_send.price = 57.49

        with patch("MetaTrader5.order_send", side_effect=[err_send, ok_send]):
            req = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": "LTCUSD",
                "volume": 1.0,
                "type": mt5.ORDER_TYPE_SELL_LIMIT,
                "price": 57.49,
                "sl": 59.11,
                "tp": 54.25,
                "type_time": mt5.ORDER_TIME_SPECIFIED,
                "expiration": 1234567,
            }
            res = te.execute_order(req)
            self.assertTrue(res["success"])
            self.assertEqual(res["ticket"], 98765)

    def test_pending_order_types_and_price_validation(self):
        """Vérifie la distinction et la validation des prix pour les 4 types d'ordres en attente."""
        from src.trading.trade_execution import TradeExecutionEngine
        from unittest.mock import MagicMock
        import MetaTrader5 as mt5

        mock_mt5 = MagicMock()
        mock_mt5.get_symbol_info.return_value = {
            "name": "EURUSD", "ask": 1.1000, "bid": 1.0998, "digits": 5, "spread": 2,
            "volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01,
            "trade_contract_size": 100000.0, "trade_tick_size": 0.00001, "trade_tick_value": 1.0,
        }
        mock_mt5.get_server_time.return_value = 1700000000.0
        te = TradeExecutionEngine(mt5_connector=mock_mt5, min_confidence=60.0)

        # 1. BUY_LIMIT : doit être strictement sous le Ask (1.1000)
        # Rejeté si entry >= ask (1.1005)
        bad_buy_limit = {"symbol": "EURUSD", "decision": {"action": "BUY", "confidence": 80, "entry": 1.1005, "sl": 1.0950, "tp": 1.1100, "order_type": "BUY_LIMIT"}}
        self.assertIsNone(te.prepare_order_request(bad_buy_limit))
        # Accepté si entry < ask (1.0990)
        good_buy_limit = {"symbol": "EURUSD", "decision": {"action": "BUY", "confidence": 80, "entry": 1.0990, "sl": 1.0950, "tp": 1.1100, "order_type": "BUY_LIMIT"}}
        req_bl = te.prepare_order_request(good_buy_limit)
        self.assertIsNotNone(req_bl)
        self.assertEqual(req_bl["type"], mt5.ORDER_TYPE_BUY_LIMIT)
        self.assertEqual(req_bl["price"], 1.0990)

        # 2. SELL_LIMIT : doit être strictement au-dessus du Bid (1.0998)
        # Rejeté si entry <= bid (1.0995)
        bad_sell_limit = {"symbol": "EURUSD", "decision": {"action": "SELL", "confidence": 80, "entry": 1.0995, "sl": 1.1050, "tp": 1.0900, "order_type": "SELL_LIMIT"}}
        self.assertIsNone(te.prepare_order_request(bad_sell_limit))
        # Accepté si entry > bid (1.1010)
        good_sell_limit = {"symbol": "EURUSD", "decision": {"action": "SELL", "confidence": 80, "entry": 1.1010, "sl": 1.1050, "tp": 1.0900, "order_type": "SELL_LIMIT"}}
        req_sl = te.prepare_order_request(good_sell_limit)
        self.assertIsNotNone(req_sl)
        self.assertEqual(req_sl["type"], mt5.ORDER_TYPE_SELL_LIMIT)
        self.assertEqual(req_sl["price"], 1.1010)

        # 3. BUY_STOP : doit être strictement au-dessus du Ask (1.1000)
        # Rejeté si entry <= ask (1.0990)
        bad_buy_stop = {"symbol": "EURUSD", "decision": {"action": "BUY", "confidence": 80, "entry": 1.0990, "sl": 1.0950, "tp": 1.1100, "order_type": "BUY_STOP"}}
        self.assertIsNone(te.prepare_order_request(bad_buy_stop))
        # Accepté si entry > ask (1.1020)
        good_buy_stop = {"symbol": "EURUSD", "decision": {"action": "BUY", "confidence": 80, "entry": 1.1020, "sl": 1.0980, "tp": 1.1100, "order_type": "BUY_STOP"}}
        req_bs = te.prepare_order_request(good_buy_stop)
        self.assertIsNotNone(req_bs)
        self.assertEqual(req_bs["type"], mt5.ORDER_TYPE_BUY_STOP)
        self.assertEqual(req_bs["price"], 1.1020)

        # 4. SELL_STOP : doit être strictement sous le Bid (1.0998)
        # Rejeté si entry >= bid (1.1010)
        bad_sell_stop = {"symbol": "EURUSD", "decision": {"action": "SELL", "confidence": 80, "entry": 1.1010, "sl": 1.1050, "tp": 1.0900, "order_type": "SELL_STOP"}}
        self.assertIsNone(te.prepare_order_request(bad_sell_stop))
        # Accepté si entry < bid (1.0980)
        good_sell_stop = {"symbol": "EURUSD", "decision": {"action": "SELL", "confidence": 80, "entry": 1.0980, "sl": 1.1020, "tp": 1.0900, "order_type": "SELL_STOP"}}
        req_ss = te.prepare_order_request(good_sell_stop)
        self.assertIsNotNone(req_ss)
        self.assertEqual(req_ss["type"], mt5.ORDER_TYPE_SELL_STOP)
        self.assertEqual(req_ss["price"], 1.0980)

    def test_execute_order_retcode_placed_and_memory_direction(self):
        """Vérifie que RETCODE_PLACED (10008) est reconnu comme succès et que BUY_LIMIT est enregistré en action BUY."""
        from src.trading.trade_execution import TradeExecutionEngine
        from unittest.mock import MagicMock, patch
        import MetaTrader5 as mt5

        mock_tm = MagicMock()
        mock_mt5 = MagicMock()
        te = TradeExecutionEngine(mt5_connector=mock_mt5, trade_memory=mock_tm)

        placed_res = MagicMock()
        placed_res.retcode = getattr(mt5, 'TRADE_RETCODE_PLACED', 10008)
        placed_res.order = 554433
        placed_res.deal = 0
        placed_res.volume = 0.5
        placed_res.price = 1.0980

        with patch("MetaTrader5.order_send", return_value=placed_res):
            req = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": "EURUSD",
                "volume": 0.5,
                "type": mt5.ORDER_TYPE_BUY_LIMIT,
                "price": 1.0980,
                "sl": 1.0940,
                "tp": 1.1060,
                "_snapshot": {"strategy": "CRT_LIMIT"},
            }
            res = te.execute_order(req)
            self.assertTrue(res["success"])
            self.assertTrue(res["is_pending"])
            self.assertEqual(res["order_type"], "BUY_LIMIT")
            self.assertEqual(res["ticket"], 554433)

            # Vérifier l'appel à record_entry
            mock_tm.record_entry.assert_called_once()
            call_kwargs = mock_tm.record_entry.call_args[1]
            self.assertEqual(call_kwargs["action"], "BUY")
            self.assertEqual(call_kwargs["snapshot"]["order_type"], "BUY_LIMIT")
            self.assertTrue(call_kwargs["snapshot"]["is_pending"])

    def test_strict_risk_ceiling_rejects_excess_min_lot(self):
        """Vérifie le respect intransigeant du plafond de risque avec strict_risk_ceiling."""
        from src.trading.risk_manager import RiskManager

        rm = RiskManager(
            default_risk_percent=0.25,
            auto_tier_enabled=True,
            strict_risk_ceiling=True,
        )
        account = {"balance": 1000.0, "margin_free": 1000.0}
        # Budget = 2.50$. Avec SL 50 pips et lot min 0.01, perte = 5.00$
        symbol = {
            "name": "EURUSD", "volume_max": 100.0, "volume_min": 0.01,
            "volume_step": 0.01, "trade_contract_size": 100000.0,
            "trade_tick_size": 0.00001, "trade_tick_value": 1.0,
            "trade_stops_level": 0, "point": 0.00001, "spread": 10,
        }
        proposal = {
            "action": "BUY",
            "entry": 1.1000,
            "sl": 1.0950,
            "tp": 1.1150,
            "confidence": 85.0
        }
        res = rm.validate_proposal(proposal, account, symbol)
        self.assertFalse(res["valid"])
        self.assertIn("Risque Intransigeant", res["reason"])

    def test_margin_capping_preserves_min_lot_when_affordable(self):
        """Vérifie que la marge libre suffisante permet d'ouvrir min_lot (0.01) sans être écrasée à 0.00."""
        from unittest.mock import patch
        from src.trading.risk_manager import RiskManager

        rm = RiskManager(fixed_lot_size=0.01)
        account = {"balance": 60.0, "margin_free": 60.0}
        symbol = {
            "name": "BTCUSD", "volume_max": 100.0, "volume_min": 0.01,
            "volume_step": 0.01, "trade_contract_size": 1.0,
            "trade_tick_size": 0.01, "trade_tick_value": 0.01,
            "trade_stops_level": 0, "point": 0.01, "spread": 50,
        }
        proposal = {
            "action": "BUY",
            "entry": 80000.0,
            "sl": 79500.0,
            "tp": 81500.0,
            "confidence": 85.0
        }
        # Cas 1: 1 lot requiert 3500$ (donc 0.01 lot requiert 35$). 60$ > 35$ => doit être accepté à 0.01 !
        with patch("MetaTrader5.order_calc_margin", return_value=3500.0):
            res1 = rm.validate_proposal(proposal, account, symbol)
            self.assertTrue(res1["valid"])
            self.assertEqual(res1["proposal"]["lot"], 0.01)

        # Cas 2: 1 lot requiert 9000$ (donc 0.01 lot requiert 90$). 60$ < 90$ => rejet propre sans crash
        with patch("MetaTrader5.order_calc_margin", return_value=9000.0):
            res2 = rm.validate_proposal(proposal, account, symbol)
            self.assertFalse(res2["valid"])
            self.assertIn("Marge libre insuffisante", res2["reason"])


if __name__ == "__main__":
    unittest.main()

