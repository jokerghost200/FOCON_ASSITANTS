import unittest
from unittest.mock import MagicMock
from src.trading.trade_execution import TradeExecutionEngine
from src.trading.trade_manager import TradeManager
import pandas as pd


class TestTradeProtection(unittest.TestCase):
    def setUp(self):
        self.mock_mt5 = MagicMock()
        self.mock_orch = MagicMock()
        self.config = {
            "trading": {
                "max_positions_per_symbol": 1,
                "break_even_enabled": True,
                "break_even_ratio": 0.45,
                "trailing_stop_enabled": True,
                "trailing_stop_atr_mult": 1.5,
                "min_profit_to_close_ratio": 0.50,
            }
        }
        self.engine = TradeExecutionEngine(
            self.mock_mt5,
            mode="auto",
            min_confidence=60.0,
            min_reward_risk=1.5,
            max_positions_per_symbol=1,
        )
        self.mock_mt5.get_daily_profit.return_value = 0.0
        self.trade_manager = TradeManager(
            self.mock_mt5,
            self.mock_orch,
            config=self.config,
        )

    def test_anti_spam_rejects_second_trade_on_same_symbol(self):
        """Astuce 3: Verrou anti-spam - 1 seul trade par symbole."""
        self.mock_mt5.get_open_positions.return_value = [
            {"ticket": 101, "symbol": "AUDCAD", "type": "BUY", "volume": 0.1}
        ]
        signal = {
            "symbol": "AUDCAD",
            "decision": {
                "action": "BUY",
                "entry": 0.9000,
                "sl": 0.8950,
                "tp": 0.9125,
                "lot": 0.1,
                "confidence": 80,
            },
        }
        # Second trade on AUDCAD must be rejected (return None)
        request = self.engine.prepare_order_request(signal)
        self.assertIsNone(request)

    def test_anti_spam_allows_trade_on_different_symbol(self):
        """Astuce 3: Autorise le trade sur un symbole different s'il n'est pas encore ouvert."""
        self.mock_mt5.get_open_positions.return_value = [
            {"ticket": 101, "symbol": "AUDCAD", "type": "BUY", "volume": 0.1}
        ]
        self.mock_mt5.get_symbol_info.return_value = {
            "ask": 1.0850,
            "bid": 1.0848,
            "spread": 2,
            "point": 0.00001,
            "trade_stops_level": 5,
        }
        signal = {
            "symbol": "EURUSD",
            "decision": {
                "action": "BUY",
                "entry": 1.0850,
                "sl": 1.0800,
                "tp": 1.0975,
                "lot": 0.1,
                "confidence": 80,
            },
        }
        request = self.engine.prepare_order_request(signal)
        self.assertIsNotNone(request)
        self.assertEqual(request["symbol"], "EURUSD")

    def test_break_even_triggers_at_45_percent_tp(self):
        """Astuce 2: Deplacement du Stop Loss a Break-Even des que 45% du TP est atteint."""
        pos = {
            "ticket": 200,
            "symbol": "EURUSD",
            "type": "BUY",
            "price_open": 1.1000,
            "price_current": 1.1050,  # 50 pips profit out of 100 pips TP (50% >= 45%)
            "sl": 1.0950,
            "tp": 1.1100,
            "profit": 50.0,
            "volume": 0.1,
        }
        self.mock_mt5.get_symbol_info.return_value = {
            "spread": 2,
            "point": 0.00001,
        }
        df = pd.DataFrame({"ATR_14": [0.0010] * 30})
        self.mock_orch.market_data.get_dataframe.return_value = df
        self.mock_orch.tech.add_all_indicators.return_value = df

        applied = self.trade_manager.apply_mechanical_protection(pos)
        self.assertTrue(applied)
        self.assertTrue(self.mock_mt5.modify_position_sl.called)
        args, kwargs = self.mock_mt5.modify_position_sl.call_args
        called_ticket, called_symbol, called_sl, called_tp = args[0], args[1], args[2], args[3]
        self.assertEqual(called_ticket, 200)
        self.assertEqual(called_symbol, "EURUSD")
        self.assertGreaterEqual(called_sl, 1.1000)

    def test_ai_close_overridden_when_profit_below_50_percent_tp(self):
        """Astuce 1: Interdiction a l'IA de couper sur bruit de spread (< 1.50$ et < 50% TP)."""
        pos = {
            "ticket": 300,
            "symbol": "EURUSD",
            "type": "BUY",
            "price_open": 1.1000,
            "price_current": 1.1005,  # Only 5 pips out of 100 pips TP (5% < 50%)
            "sl": 1.0950,
            "tp": 1.1100,
            "profit": 0.80,  # Bruit de spread < 1.50$
            "volume": 0.1,
        }
        self.mock_mt5.get_open_positions.return_value = [pos]
        self.mock_mt5.get_symbol_info.return_value = {
            "spread": 2,
            "point": 0.00001,
        }
        df = pd.DataFrame({"ATR_14": [0.0010] * 30})
        self.mock_orch.market_data.get_dataframe.return_value = df
        self.mock_orch.tech.add_all_indicators.return_value = df
        self.mock_orch.analyze_open_position.return_value = {
            "action": "CLOSE",
            "reason": "Micro spread cut",
        }

        self.trade_manager.manage_open_positions(auto_mode="auto")
        self.mock_mt5.close_position.assert_not_called()

    def test_dollar_profit_lock_drawdown_large(self):
        """Verrou Dollars: Si le profit monte a 17$ et reflue de >= 3$ (ex: 13.50$), cloture immediate."""
        pos = {
            "ticket": 401,
            "symbol": "XAUUSD",
            "type": "BUY",
            "price_open": 2000.0,
            "price_current": 2013.5,
            "profit": 13.50,
            "volume": 0.01,
        }
        # Pre-seed historical peak at 17.00$
        self.trade_manager.peak_profits[401] = 17.00
        self.mock_mt5.close_position.return_value = True

        closed = self.trade_manager.check_dollar_profit_lock(pos)
        self.assertTrue(closed)
        self.mock_mt5.close_position.assert_called_once_with(401, "XAUUSD", "BUY", 0.01, 2013.5)

    def test_dollar_profit_lock_drawdown_medium(self):
        """Verrou Dollars: Si le profit monte a 10$ et reflue de >= 1.8$ (ex: 8.00$), cloture immediate."""
        pos = {
            "ticket": 402,
            "symbol": "GBPJPY",
            "type": "BUY",
            "price_open": 180.0,
            "price_current": 180.8,
            "profit": 8.00,
            "volume": 0.01,
        }
        self.trade_manager.peak_profits[402] = 10.00
        self.mock_mt5.close_position.return_value = True

        closed = self.trade_manager.check_dollar_profit_lock(pos)
        self.assertTrue(closed)
        self.mock_mt5.close_position.assert_called_once_with(402, "GBPJPY", "BUY", 0.01, 180.8)

    def test_daily_profit_target_closes_winners_and_stops(self):
        """Objectif journalier: Si gain du jour >= 50$, cloture des positions gagnantes pour verrouiller."""
        self.trade_manager.daily_profit_target_usd = 50.0
        self.mock_mt5.get_daily_profit.return_value = 52.50
        winner_pos = {
            "ticket": 501,
            "symbol": "EURUSD",
            "type": "BUY",
            "price_open": 1.0800,
            "price_current": 1.0850,
            "profit": 15.00,
            "volume": 0.05,
        }
        loser_pos = {
            "ticket": 502,
            "symbol": "USDCAD",
            "type": "SELL",
            "price_open": 1.3500,
            "price_current": 1.3510,
            "profit": -4.00,
            "volume": 0.05,
        }
        self.mock_mt5.get_open_positions.return_value = [winner_pos, loser_pos]

        self.trade_manager.manage_open_positions(auto_mode="auto")
        # Winner must be closed, loser left to hit SL / protected
        self.mock_mt5.close_position.assert_called_once_with(501, "EURUSD", "BUY", 0.05, 1.0850)

    def test_engine_blocks_new_orders_when_daily_target_reached(self):
        """Moteur d'ordre: Rejette toute nouvelle entree si l'objectif journalier est deja atteint."""
        self.engine.daily_profit_target_usd = 50.0
        self.mock_mt5.get_daily_profit.return_value = 55.00
        signal = {
            "symbol": "EURUSD",
            "decision": {
                "action": "BUY",
                "entry": 1.0850,
                "sl": 1.0800,
                "tp": 1.0975,
                "lot": 0.05,
                "confidence": 85,
            },
        }
        request = self.engine.prepare_order_request(signal)
        self.assertIsNone(request)

    def test_anti_fomo_rsi_overbought_overrides_to_wait(self):
        """Amélioration Entrée: Interdiction formelle d'acheter sur sommet (RSI >= 70)."""
        from src.ai.orchestrator import AIOrchestrator
        orch = AIOrchestrator(
            mt5_connector=self.mock_mt5,
            market_data=MagicMock(),
            tech_indicators=MagicMock(),
            ai_router=MagicMock(),
            risk_manager=MagicMock(),
            live_news_enabled=False,
            order_block_scan_enabled=False,
        )
        # Mock dataframe and features with RSI = 75 (Overbought)
        df = pd.DataFrame({"close": [1.1000] * 30, "open": [1.0990] * 30, "high": [1.1010] * 30, "low": [1.0980] * 30})
        orch.market_data.get_dataframe.return_value = df
        orch.tech.add_all_indicators.return_value = df
        orch.tech.extract_latest_features.return_value = {
            "close": 1.1000,
            "ema_50": 1.0950,
            "ema_200": 1.0900,
            "rsi_14": 75.0,  # OVERBOUGHT!
            "atr_14": 0.0010,
            "trend": "BULLISH",
        }
        orch.market_data.analyze_amd.return_value = {"Phase": "Expansion", "FVG": "None", "BOS_MSS": "None", "Liquidity_Grab": "None"}
        self.mock_mt5.get_symbol_info.return_value = {"spread": 2, "point": 0.00001, "trade_stops_level": 5}
        orch.market_data.get_market_depth.return_value = {"available": False}
        
        # AI foolishly outputs BUY chasing the green pump
        orch.router.ask_text.return_value = '{"action": "BUY", "confidence": 85, "entry": 1.1000, "sl": 1.0950, "tp": 1.1125, "reason": "Strong pump"}'

        result = orch.analyze_symbol("EURUSD", "M15")
        self.assertEqual(result.get("status"), "WAIT")
        self.assertEqual(result["decision"]["action"], "WAIT")
        self.assertIn("Anti-FOMO", result["decision"]["reason"])

    def test_anti_chasing_overextended_overrides_to_wait(self):
        """Amélioration Entrée: Interdiction d'entrer si le cours est sur-étendu (> 2.8 ATR de l'EMA 50)."""
        from src.ai.orchestrator import AIOrchestrator
        orch = AIOrchestrator(
            mt5_connector=self.mock_mt5,
            market_data=MagicMock(),
            tech_indicators=MagicMock(),
            ai_router=MagicMock(),
            risk_manager=MagicMock(),
            live_news_enabled=False,
            order_block_scan_enabled=False,
        )
        # Mock price 4 ATR above EMA 50
        df = pd.DataFrame({"close": [1.1000] * 30, "open": [1.0990] * 30, "high": [1.1010] * 30, "low": [1.0980] * 30})
        orch.market_data.get_dataframe.return_value = df
        orch.tech.add_all_indicators.return_value = df
        orch.tech.extract_latest_features.return_value = {
            "close": 1.1000,
            "ema_50": 1.0950,  # 50 pips below = 5 ATR distance
            "ema_200": 1.0900,
            "rsi_14": 62.0,
            "atr_14": 0.0010,
            "trend": "BULLISH",
        }
        orch.market_data.analyze_amd.return_value = {"Phase": "Expansion", "FVG": "None", "BOS_MSS": "None", "Liquidity_Grab": "None"}
        self.mock_mt5.get_symbol_info.return_value = {"spread": 2, "point": 0.00001, "trade_stops_level": 5}
        orch.market_data.get_market_depth.return_value = {"available": False}
        
        orch.router.ask_text.return_value = '{"action": "BUY", "confidence": 80, "entry": 1.1000, "sl": 1.0950, "tp": 1.1125, "reason": "Chasing trend"}'

        result = orch.analyze_symbol("EURUSD", "M15")
        self.assertEqual(result.get("status"), "WAIT")
        self.assertEqual(result["decision"]["action"], "WAIT")
        self.assertIn("Anti-Chasing", result["decision"]["reason"])

    def test_local_quant_brain_offline_fallback_patient_wait_without_confluence(self):
        """Cerveau local: Si l'API est hors-ligne et sans confluence, le cerveau local attend sagement (WAIT)."""
        from src.ai.orchestrator import AIOrchestrator
        orch = AIOrchestrator(
            mt5_connector=self.mock_mt5,
            market_data=MagicMock(),
            tech_indicators=MagicMock(),
            ai_router=MagicMock(),
            risk_manager=MagicMock(),
            live_news_enabled=False,
            order_block_scan_enabled=False,
        )
        df = pd.DataFrame({"close": [1.1000] * 30, "open": [1.0990] * 30, "high": [1.1010] * 30, "low": [1.0980] * 30})
        orch.market_data.get_dataframe.return_value = df
        orch.tech.add_all_indicators.return_value = df
        orch.tech.extract_latest_features.return_value = {
            "close": 1.1000,
            "ema_50": 1.0980,
            "ema_200": 1.0950,
            "rsi_14": 52.0,
            "atr_14": 0.0010,
            "trend": "BULLISH",
        }
        orch.market_data.analyze_amd.return_value = {"Phase": "Expansion", "FVG": "None", "BOS_MSS": "None", "Liquidity_Grab": "None"}
        self.mock_mt5.get_symbol_info.return_value = {"spread": 2, "point": 0.00001, "trade_stops_level": 5}
        orch.market_data.get_market_depth.return_value = {"available": False}
        
        # Simulate API failure / quota limit (returns None)
        orch.router.ask_text.return_value = None

        result = orch.analyze_symbol("EURUSD", "M15")
        self.assertIsNotNone(result)
        self.assertEqual(result.get("status"), "WAIT")
        self.assertIn("Cerveau", result["decision"]["reason"])

    def test_local_quant_brain_offline_fallback_signal_with_confluence(self):
        """Cerveau local: Si l'API est hors-ligne mais avec confluence (FVG + Boomerang), génère un signal BUY."""
        from src.ai.orchestrator import AIOrchestrator
        orch = AIOrchestrator(
            mt5_connector=self.mock_mt5,
            market_data=MagicMock(),
            tech_indicators=MagicMock(),
            ai_router=MagicMock(),
            risk_manager=MagicMock(),
            live_news_enabled=False,
            order_block_scan_enabled=False,
        )
        df = pd.DataFrame({"close": [1.1000] * 30, "open": [1.0990] * 30, "high": [1.1010] * 30, "low": [1.0980] * 30})
        orch.market_data.get_dataframe.return_value = df
        orch.tech.add_all_indicators.return_value = df
        orch.tech.extract_latest_features.return_value = {
            "close": 1.1000,
            "ema_50": 1.0980,
            "ema_200": 1.0950,
            "rsi_14": 52.0,
            "atr_14": 0.0010,
            "macd_hist": 0.0004,
            "trend": "BULLISH",
        }
        # Confluence: FVG Bullish + Boomerang BUY
        orch.market_data.analyze_amd.return_value = {
            "Phase": "Expansion",
            "FVG": "Bullish FVG",
            "BOS_MSS": "BOS Bullish",
            "Liquidity_Grab": "None",
        }
        mock_boom = MagicMock()
        mock_boom.decision = "VALID"
        mock_boom.direction = "BUY"
        mock_boom.score = 85.0
        mock_boom.summary.return_value = "Boomerang VALID BUY"
        orch.boomerang.detect = MagicMock(return_value=mock_boom)

        self.mock_mt5.get_symbol_info.return_value = {
            "spread": 2,
            "point": 0.00001,
            "trade_stops_level": 5,
            "ask": 1.1001,
            "bid": 1.0999,
        }
        self.mock_mt5.get_account_info.return_value = {"balance": 1000.0, "equity": 1000.0}
        orch.market_data.get_market_depth.return_value = {"available": False}
        orch.risk.validate_proposal.return_value = {
            "valid": True,
            "proposal": {
                "action": "BUY",
                "entry": 1.1000,
                "sl": 1.0950,
                "tp": 1.1125,
                "lot": 0.05,
                "confidence": 75.0,
                "reason": "Cerveau Quant Local: Signal confirmé",
            },
        }

        # Simulate API failure / quota limit
        orch.router.ask_text.return_value = None

        result = orch.analyze_symbol("EURUSD", "M15")
        self.assertIsNotNone(result)
        self.assertEqual(result.get("status"), "SIGNAL")
        self.assertEqual(result["decision"]["action"], "BUY")
        self.assertIn("Cerveau", result["decision"]["reason"])
        self.assertGreaterEqual(result["decision"]["confidence"], 65.0)

    def test_take_profit_target_per_trade_usd_triggers_immediate_close(self):
        """Vérifie que la position est immédiatement coupée dès qu'elle atteint l'objectif fixe en $ (ex: 4.00 $)."""
        self.trade_manager.take_profit_target_per_trade_usd = 4.0
        pos = {"ticket": 901, "symbol": "EURUSD", "type": "BUY", "profit": 4.02, "price_open": 1.1000}
        
        # When profit reaches $4.02 >= $4.00, it should return True (close position)
        should_close = self.trade_manager.check_dollar_profit_lock(pos)
        self.assertTrue(should_close)
        self.mock_mt5.close_position.assert_called()
        self.assertEqual(self.mock_mt5.close_position.call_args[0][0], 901)

    def test_take_profit_target_per_trade_usd_does_not_close_before_target(self):
        """Vérifie que la position n'est pas coupée tant qu'elle est sous le seuil fixe en $ (ex: 3.50 $ < 4.00 $)."""
        self.trade_manager.take_profit_target_per_trade_usd = 4.0
        self.mock_mt5.reset_mock()
        pos = {"ticket": 902, "symbol": "EURUSD", "type": "BUY", "profit": 3.50, "price_open": 1.1000}
        
        should_close = self.trade_manager.check_dollar_profit_lock(pos)
        self.assertFalse(should_close)
        self.mock_mt5.close_position.assert_not_called()

    def test_allowed_trade_direction_buy_only_blocks_sell(self):
        """Vérifie que le mode BUY_ONLY bloque strictement les ordres SELL."""
        self.engine.allowed_trade_direction = "BUY_ONLY"
        self.mock_mt5.get_open_positions.return_value = []
        self.mock_mt5.get_symbol_info.return_value = {
            "ask": 1.0850, "bid": 1.0848, "spread": 2, "point": 0.00001, "trade_stops_level": 5,
        }
        sell_signal = {
            "symbol": "EURUSD",
            "decision": {
                "action": "SELL", "entry": 1.0848, "sl": 1.0900, "tp": 1.0700, "lot": 0.1, "confidence": 80,
            },
        }
        request = self.engine.prepare_order_request(sell_signal)
        self.assertIsNone(request)

        buy_signal = {
            "symbol": "EURUSD",
            "decision": {
                "action": "BUY", "entry": 1.0850, "sl": 1.0800, "tp": 1.0950, "lot": 0.1, "confidence": 80,
            },
        }
        request_buy = self.engine.prepare_order_request(buy_signal)
        self.assertIsNotNone(request_buy)
        import MetaTrader5 as mt5
        self.assertEqual(request_buy["type"], mt5.ORDER_TYPE_BUY)

    def test_allowed_trade_direction_sell_only_blocks_buy(self):
        """Vérifie que le mode SELL_ONLY bloque strictement les ordres BUY."""
        self.engine.allowed_trade_direction = "SELL_ONLY"
        self.mock_mt5.get_open_positions.return_value = []
        self.mock_mt5.get_symbol_info.return_value = {
            "ask": 1.0850, "bid": 1.0848, "spread": 2, "point": 0.00001, "trade_stops_level": 5,
        }
        buy_signal = {
            "symbol": "EURUSD",
            "decision": {
                "action": "BUY", "entry": 1.0850, "sl": 1.0800, "tp": 1.0950, "lot": 0.1, "confidence": 80,
            },
        }
        request = self.engine.prepare_order_request(buy_signal)
        self.assertIsNone(request)

        sell_signal = {
            "symbol": "EURUSD",
            "decision": {
                "action": "SELL", "entry": 1.0848, "sl": 1.0900, "tp": 1.0700, "lot": 0.1, "confidence": 80,
            },
        }
        request_sell = self.engine.prepare_order_request(sell_signal)
        self.assertIsNotNone(request_sell)
        import MetaTrader5 as mt5
        self.assertEqual(request_sell["type"], mt5.ORDER_TYPE_SELL)

    def test_local_quant_brain_direction_filtering(self):
        """Vérifie que le cerveau local quantitatif respecte le filtre de direction."""
        from src.ai.local_quant_brain import LocalQuantBrain
        brain = LocalQuantBrain(min_confidence=65.0)
        indicators = {
            "close": 100.0,
            "rsi_14": 50.0,  # healthy pullback zone
            "ema_50": 98.0,
            "ema_200": 90.0,
            "macd": 0.5,
            "macd_signal": 0.2,
            "macd_hist": 0.3,
            "adx": 30.0,
            "atr_14": 1.5,
        }
        boomerang_data = {
            "decision": "VALID",
            "direction": "BUY",
            "score": 80.0,
        }
        # In SELL_ONLY, bull_score is zeroed out, resulting in WAIT
        decision = brain.decide(
            "BTCUSD", "M15", indicators,
            boomerang_data=boomerang_data,
            htf_trend="BULLISH",
            direction_filter="SELL_ONLY",
        )
        self.assertEqual(decision["action"], "WAIT")

        # In BUY_ONLY, bull decision should be allowed
        decision_buy = brain.decide(
            "BTCUSD", "M15", indicators,
            boomerang_data=boomerang_data,
            htf_trend="BULLISH",
            direction_filter="BUY_ONLY",
        )
        self.assertEqual(decision_buy["action"], "BUY")

    def test_manage_open_positions_skips_closed_market(self):
        """Vérifie que les positions sur marchés fermés sont ignorées (pas d'appel IA ni modif SL)."""
        pos = {
            "ticket": 501,
            "symbol": "AUDJPY",
            "type": "SELL",
            "price_open": 110.0,
            "price_current": 109.5,
            "profit": 45.0,
            "volume": 0.1,
            "sl": 111.0,
            "tp": 108.0,
        }
        self.mock_mt5.get_open_positions.return_value = [pos]
        self.mock_mt5.is_market_open.return_value = False

        self.trade_manager.manage_open_positions(auto_mode="auto")

        # Ne doit ni tenter de fermer, ni de modifier le SL, ni d'appeler l'IA
        self.mock_mt5.close_position.assert_not_called()
        self.mock_mt5.modify_position_sl.assert_not_called()
        self.mock_orch.analyze_open_position.assert_not_called()

    def test_manage_open_positions_skips_unallowed_market_types(self):
        """Vérifie que les positions dont le type de marché est désactivé sont ignorées."""
        self.trade_manager.config["trading"]["allowed_market_types"] = ["Crypto"]
        pos = {
            "ticket": 502,
            "symbol": "EURUSD",
            "type": "BUY",
            "price_open": 1.10,
            "price_current": 1.105,
            "profit": 50.0,
            "volume": 0.1,
            "sl": 1.09,
            "tp": 1.12,
        }
        self.mock_mt5.get_open_positions.return_value = [pos]
        self.mock_mt5.is_market_open.return_value = True
        self.mock_mt5.is_symbol_in_allowed_types.return_value = False

        self.trade_manager.manage_open_positions(auto_mode="auto")

        self.mock_mt5.close_position.assert_not_called()
        self.mock_mt5.modify_position_sl.assert_not_called()
        self.mock_orch.analyze_open_position.assert_not_called()

    def test_manage_open_positions_skips_ai_when_profit_negative(self):
        """Vérifie que l'IA externe n'est pas appelée quand profit < 0 (SL actif sur broker)."""
        pos = {
            "ticket": 601,
            "symbol": "BTCUSD",
            "type": "SELL",
            "price_open": 84000.0,
            "price_current": 84010.0,
            "profit": -9.92,  # négatif (spread / léger retracement)
            "volume": 0.52,
            "sl": 84210.0,
            "tp": 83500.0,
        }
        self.mock_mt5.get_open_positions.return_value = [pos]
        self.mock_mt5.is_market_open.return_value = True
        self.mock_mt5.is_symbol_in_allowed_types.return_value = True

        self.trade_manager.manage_open_positions(auto_mode="auto")

        # L'IA externe ne doit PAS être appelée
        self.mock_orch.analyze_open_position.assert_not_called()

    def test_manage_open_positions_throttles_ai_calls(self):
        """Vérifie que l'évaluation IA par position est throttlée (au plus 1 fois par intervalle)."""
        pos = {
            "ticket": 602,
            "symbol": "LNKUSD",
            "type": "BUY",
            "price_open": 14.20,
            "price_current": 14.30,
            "profit": 5.0,  # positif
            "volume": 1.0,
            "sl": 14.07,
            "tp": 14.80,
        }
        self.mock_mt5.get_open_positions.return_value = [pos]
        self.mock_mt5.is_market_open.return_value = True
        self.mock_mt5.is_symbol_in_allowed_types.return_value = True
        self.mock_orch.analyze_open_position.return_value = {"action": "HOLD", "reason": "Trend valid"}

        # 1er appel -> L'IA doit être appelée
        self.trade_manager.manage_open_positions(auto_mode="auto")
        self.assertEqual(self.mock_orch.analyze_open_position.call_count, 1)

        # 2ème appel immédiat (même seconde) -> L'IA doit être throttlée (pas de 2ème appel)
        self.trade_manager.manage_open_positions(auto_mode="auto")
        self.assertEqual(self.mock_orch.analyze_open_position.call_count, 1)

    def test_effective_daily_profit_target_percentage(self):
        """Vérifie que l'objectif journalier s'adapte en % du capital (ex: 3% de 2000$ = 60$)."""
        tm = TradeManager(
            self.mock_mt5,
            self.mock_orch,
            config={
                "trading": {
                    "daily_profit_target": 50.0,
                    "daily_profit_target_percent": 3.0,
                }
            },
        )
        self.mock_mt5.get_account_info.return_value = {"balance": 2000.0}
        eff_target = tm.get_effective_daily_profit_target()
        self.assertAlmostEqual(eff_target, 60.0)

        # Si daily_profit_target_percent est 0 ou None, repli sur la valeur fixe en $
        tm.daily_profit_target_percent = 0.0
        eff_target_fixed = tm.get_effective_daily_profit_target()
        self.assertAlmostEqual(eff_target_fixed, 50.0)

    def test_dollar_profit_lock_adapts_to_micro_tier(self):
        """Vérifie que pour un micro-compte (< 250$), le seuil de protection dollar descend à 1.00$ au lieu de 3.00$."""
        tm = TradeManager(
            self.mock_mt5,
            self.mock_orch,
            config={
                "trading": {
                    "auto_tier_enabled": True,
                    "dollar_profit_lock_enabled": True,
                    "min_profit_to_protect": 3.0,
                    "max_profit_giveback_dollar": 1.0,
                }
            },
        )
        self.mock_mt5.get_account_info.return_value = {"balance": 150.0}  # Micro tier

        pos = {
            "ticket": 701,
            "symbol": "EURUSD",
            "type": "BUY",
            "profit": 1.20,  # > 1.00$ (palier micro), mais < 3.00$ (seuil standard)
            "price_current": 1.1020,
        }

        # Simule un pic à 1.80 $ puis une baisse à 0.70 $ (giveback de 1.10 $ > 1.00 $)
        tm.peak_profits[701] = 1.80
        pos["profit"] = 0.70
        closed = tm.check_dollar_profit_lock(pos)
        self.assertTrue(closed)
        self.mock_mt5.close_position.assert_called_once_with(701, "EURUSD", "BUY", 0.01, 1.1020)

    def test_ob_entry_drift_preserves_reward_risk_without_rejection(self):
        """Vérifie que lors d'un léger drift à l'entrée sur OB, le TP recalculé n'est pas rejeté pour R:R < 2.50."""
        from src.trading.trade_execution import TradeExecutionEngine
        te = TradeExecutionEngine(self.mock_mt5, min_reward_risk=2.50, min_confidence=70.0)
        
        # Simule le cas NZDCHF des logs utilisateur
        self.mock_mt5.get_symbol_info.return_value = {
            "name": "NZDCHF", "digits": 5, "ask": 0.46904, "bid": 0.46900,
            "trade_tick_size": 0.00001, "trade_tick_value": 1.0,
            "volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01,
            "filling_mode": 1
        }
        signal = {
            "status": "SIGNAL",
            "symbol": "NZDCHF",
            "decision": {
                "action": "BUY",
                "entry": 0.46878,
                "sl": 0.46798,
                "tp": 0.47078,  # R:R original = (0.47078 - 0.46878) / (0.46878 - 0.46798) = 2.50
                "confidence": 80.0,
                "lot": 0.05
            }
        }
        # Entrée réelle au marché à 0.46904 (drift de 0.46878 -> 0.46904)
        req = te.prepare_order_request(signal)
        self.assertIsNotNone(req, "La requête ne doit pas être rejetée pour un léger arrondi de R:R !")
        self.assertEqual(req["symbol"], "NZDCHF")
        self.assertEqual(req["volume"], 0.05)


if __name__ == "__main__":
    unittest.main()




