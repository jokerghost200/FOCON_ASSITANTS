import unittest
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from unittest.mock import MagicMock

from src.core.crt_detector import CRTDetector, CRTSetup, NY_TZ
from src.trading.strategy_engine import CRTStrategy, StrategySelector


class TestCRTDetector(unittest.TestCase):

    def setUp(self):
        self.detector = CRTDetector(min_sweep_atr_mult=0.1, min_range_atr_mult=0.5, min_rr_tp1=1.5)

    def test_bullish_crt_valid(self):
        """
        Exemple du guide :
        C1 (H4) : Low 1.0950, High 1.1000, Close 1.0980 (Range = 50 pips)
        C2 (H4) : Low 1.0940 (sweep de 10 pips), Close 1.0968 (dans le range de C1), High 1.0990
        C3 (H4 en cours) : 1.0970
        """
        df = pd.DataFrame([
            {"open": 1.0960, "high": 1.1000, "low": 1.0950, "close": 1.0980},  # C1 (iloc[-3])
            {"open": 1.0980, "high": 1.0990, "low": 1.0940, "close": 1.0968},  # C2 (iloc[-2])
            {"open": 1.0968, "high": 1.0975, "low": 1.0965, "close": 1.0970},  # C3 (iloc[-1] en cours)
        ])
        atr = 0.0050  # 50 pips ATR

        setup = self.detector.detect(df, atr=atr)
        self.assertIsNotNone(setup)
        self.assertEqual(setup.direction, "BUY")
        self.assertEqual(setup.c1_low, 1.0950)
        self.assertEqual(setup.c1_high, 1.1000)
        self.assertEqual(setup.c1_mid, 1.0975)
        self.assertEqual(setup.sl_raw, 1.0940)
        self.assertEqual(setup.tp1, 1.0975)
        self.assertEqual(setup.tp2, 1.1000)
        self.assertAlmostEqual(setup.sweep_depth, 0.0010)

    def test_bearish_crt_valid(self):
        """
        Exemple baissier du guide :
        C1 (H4) : Low 1.0950, High 1.1000 (Range = 50 pips)
        C2 (H4) : High 1.1012 (sweep de 12 pips), Close 1.0982 (dans le range), Low 1.0960
        C3 en cours
        """
        df = pd.DataFrame([
            {"open": 1.0960, "high": 1.1000, "low": 1.0950, "close": 1.0970},  # C1
            {"open": 1.0970, "high": 1.1012, "low": 1.0960, "close": 1.0982},  # C2
            {"open": 1.0982, "high": 1.0985, "low": 1.0975, "close": 1.0980},  # C3
        ])
        atr = 0.0050

        setup = self.detector.detect(df, atr=atr)
        self.assertIsNotNone(setup)
        self.assertEqual(setup.direction, "SELL")
        self.assertEqual(setup.c1_high, 1.1000)
        self.assertEqual(setup.c1_low, 1.0950)
        self.assertEqual(setup.c1_mid, 1.0975)
        self.assertEqual(setup.sl_raw, 1.1012)
        self.assertEqual(setup.tp1, 1.0975)
        self.assertEqual(setup.tp2, 1.0950)
        self.assertAlmostEqual(setup.sweep_depth, 0.0012)

    def test_rejection_c2_closes_outside_range(self):
        """Si C2 clôture hors du range (cassure réelle), le setup est invalidé."""
        df = pd.DataFrame([
            {"open": 1.0960, "high": 1.1000, "low": 1.0950, "close": 1.0970},  # C1
            {"open": 1.0970, "high": 1.0980, "low": 1.0940, "close": 1.0935},  # C2 close SOUS C1 Low
            {"open": 1.0935, "high": 1.0940, "low": 1.0930, "close": 1.0938},  # C3
        ])
        setup = self.detector.detect(df, atr=0.0050)
        self.assertIsNone(setup)

    def test_rejection_double_sweep_ambiguous(self):
        """Si C2 balaie à la fois le High ET le Low de C1, signal ambigu rejeté."""
        df = pd.DataFrame([
            {"open": 1.0960, "high": 1.1000, "low": 1.0950, "close": 1.0970},  # C1
            {"open": 1.0970, "high": 1.1015, "low": 1.0935, "close": 1.0975},  # C2 dépasse les deux côtés
            {"open": 1.0975, "high": 1.0980, "low": 1.0970, "close": 1.0972},  # C3
        ])
        setup = self.detector.detect(df, atr=0.0050)
        self.assertIsNone(setup)

    def test_rejection_shallow_sweep(self):
        """Filtre de bruit : sweep inférieur au minimum requis d'ATR."""
        df = pd.DataFrame([
            {"open": 1.0960, "high": 1.1000, "low": 1.0950, "close": 1.0970},  # C1
            {"open": 1.0970, "high": 1.0990, "low": 1.09495, "close": 1.0965}, # Sweep de seulement 0.5 pip
            {"open": 1.0965, "high": 1.0970, "low": 1.0960, "close": 1.0965},
        ])
        atr = 0.0050  # 50 pips ATR -> min_sweep = 5 pips (0.0005)
        setup = self.detector.detect(df, atr=atr)
        self.assertIsNone(setup)

    def test_killzone_detection_with_zoneinfo(self):
        """Vérifie la détection des Killzones basées sur l'heure de New York."""
        # 1. Killzone Londres : 03:00 heure de NY (07:00 UTC)
        dt_london = datetime(2026, 10, 22, 7, 30, tzinfo=ZoneInfo("UTC"))
        res_london = CRTDetector.check_killzone(dt_london)
        self.assertTrue(res_london["in_killzone"])
        self.assertEqual(res_london["name"], "London")

        # 2. Killzone New York : 08:30 heure de NY (12:30 UTC)
        dt_ny = datetime(2026, 10, 22, 12, 30, tzinfo=ZoneInfo("UTC"))
        res_ny = CRTDetector.check_killzone(dt_ny)
        self.assertTrue(res_ny["in_killzone"])
        self.assertEqual(res_ny["name"], "New York")

        # 3. Hors Killzone : 15:00 heure de NY (19:00 UTC)
        dt_off = datetime(2026, 10, 22, 19, 0, tzinfo=ZoneInfo("UTC"))
        res_off = CRTDetector.check_killzone(dt_off)
        self.assertFalse(res_off["in_killzone"])

    def test_crt_strategy_evaluation(self):
        """Vérifie l'évaluation de CRTStrategy dans le moteur de trading."""
        crt_setup = CRTSetup(
            direction="BUY",
            c1_high=1.1000,
            c1_low=1.0950,
            c1_mid=1.0975,
            c2_high=1.0980,
            c2_low=1.0940,
            c2_close=1.0955,
            sweep_price=1.0940,
            sweep_depth=0.0010,
            sl_raw=1.0940,
            tp1=1.0975,
            tp2=1.1000,
            in_killzone=True,
            killzone_name="London"
        )

        strat = CRTStrategy(min_rr_tp1=1.2)
        tech_features = {"atr_14": 0.0010}
        amd_context = {"BOS_MSS": "Bullish MSS", "FVG": "Bullish FVG"}
        price = 1.0950  # Entrée optimale sur retest
        spread = 0.0001
        min_distance = 0.0002

        proposal = strat.evaluate(
            tech_features=tech_features,
            amd_context=amd_context,
            htf_trend="BULLISH",
            price=price,
            spread=spread,
            min_distance=min_distance,
            crt_setup=crt_setup
        )

        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["direction"], "BUY")
        self.assertLess(proposal["sl"], crt_setup.sl_raw) # SL intègre le buffer
        self.assertEqual(proposal["tp"], 1.1000)          # Borne haute
        self.assertEqual(proposal["tp1"], 1.0975)         # 50% EQ
        self.assertTrue(proposal["in_killzone"])
        self.assertGreaterEqual(proposal["rr_tp1"], 1.2)

    def test_strategy_selector_with_crt(self):
        """Vérifie que StrategySelector active CRT en régime RANGE ou REVERSAL."""
        selector = StrategySelector()
        crt_setup = CRTSetup(
            direction="SELL",
            c1_high=1.1000,
            c1_low=1.0950,
            c1_mid=1.0975,
            c2_high=1.1010,
            c2_low=1.0970,
            c2_close=1.0985,
            sweep_price=1.1010,
            sweep_depth=0.0010,
            sl_raw=1.1010,
            tp1=1.0975,
            tp2=1.0950,
            in_killzone=True,
            killzone_name="New York"
        )
        tech_features = {"atr_14": 0.0010}
        amd_context = {"BOS_MSS": "Bearish MSS"}
        price = 1.1002  # Entrée haute après retest (sl=1.1012, risk=0.0010, dist to tp1=0.0027 -> R:R 2.7)

        proposals = selector.select_and_evaluate(
            market_regime="REVERSAL",
            tech_features=tech_features,
            amd_context=amd_context,
            htf_trend="BEARISH",
            price=price,
            spread=0.0001,
            min_distance=0.0002,
            crt_setup=crt_setup
        )

        # Doit contenir une proposition CRT
        crt_prop = next((p for p in proposals if "CRT" in p["strategy"]), None)
        self.assertIsNotNone(crt_prop)
        self.assertEqual(crt_prop["direction"], "SELL")
        self.assertEqual(crt_prop["tp"], 1.0950)

    def test_local_brain_with_crt_setup(self):
        """Vérifie que LocalQuantBrain intègre le setup CRT dans sa décision et ses niveaux."""
        from src.ai.local_quant_brain import LocalQuantBrain

        brain = LocalQuantBrain(min_confidence=50.0, min_rr=1.5)
        crt_setup = CRTSetup(
            direction="BUY",
            c1_high=1.1000,
            c1_low=1.0950,
            c1_mid=1.0975,
            c2_high=1.0980,
            c2_low=1.0940,
            c2_close=1.0965,
            sweep_price=1.0940,
            sweep_depth=0.0010,
            sl_raw=1.0940,
            tp1=1.0975,
            tp2=1.1000,
            in_killzone=True,
            killzone_name="London"
        )
        tech_features = {
            "close": 1.0965,
            "ema_50": 1.0960,
            "ema_200": 1.0950,
            "rsi_14": 52.0,
            "atr_14": 0.0010,
            "trend": "BULLISH",
        }
        res = brain.decide(
            symbol="EURUSD",
            timeframe="M15",
            tech_features=tech_features,
            htf_trend="BULLISH",
            spread_price=0.0001,
            min_distance=0.0002,
            crt_setup=crt_setup
        )

        self.assertEqual(res["action"], "BUY")
        self.assertIn("CRT Bullish", res["reason"])
        self.assertEqual(res["tp"], 1.1000)
        self.assertLessEqual(res["sl"], 1.0940)

    def test_smc_pending_bullish_valid(self):
        """
        Guide Universel Section 3, 4 et 7 :
        HTF Range : C1 High = 1.1000, Low = 1.0950, Mid = 1.0975
        LTF :
          - Swings confirmés avec swing_lookback=2
          - Sweep sous 1.0950 (Low = 1.0940)
          - Réintégration close > 1.0950
          - MSS impulsif au-dessus du dernier Swing High
          - FVG créé par le déplacement
          - Ordre BUY_LIMIT généré au milieu du FVG
        """
        df_htf = pd.DataFrame([
            {"open": 1.0960, "high": 1.0990, "low": 1.0950, "close": 1.0970},
            {"open": 1.0970, "high": 1.1000, "low": 1.0950, "close": 1.0980},  # C1 clôturé (iloc[-2])
            {"open": 1.0980, "high": 1.0985, "low": 1.0975, "close": 1.0980},  # Bougie active
        ])

        # Indices 0..4 créent un swing high à l'indice 2 (lookback=2)
        # index 2: high=1.0970, confirm_idx = 4
        # index 5: sweep low = 1.0940 (sous 1.0950)
        # index 6: réintégration close = 1.0954 (> 1.0950)
        # index 7: displacement bougie 1 -> high = 1.0958, low = 1.0952
        # index 8: displacement bougie 2 -> high = 1.0968, low = 1.0958
        # index 9: displacement bougie 3 -> low = 1.0965 (> 1.0958 donc FVG=[1.0958, 1.0965]), close = 1.0972 (> 1.0970 donc MSS!)
        # index 10: bougie active (en cours)
        df_ltf = pd.DataFrame([
            {"open": 1.0955, "high": 1.0958, "low": 1.0952, "close": 1.0955},
            {"open": 1.0955, "high": 1.0960, "low": 1.0953, "close": 1.0958},
            {"open": 1.0958, "high": 1.0970, "low": 1.0955, "close": 1.0965},  # 2: SWING HIGH
            {"open": 1.0965, "high": 1.0962, "low": 1.0952, "close": 1.0955},
            {"open": 1.0955, "high": 1.0958, "low": 1.0948, "close": 1.0952},  # 4: CONFIRMATION SWING
            {"open": 1.0952, "high": 1.0952, "low": 1.0940, "close": 1.0945},  # 5: SWEEP
            {"open": 1.0945, "high": 1.0955, "low": 1.0944, "close": 1.0954},  # 6: REINTEGRATION
            {"open": 1.0954, "high": 1.0958, "low": 1.0951, "close": 1.0956},  # 7: B1 du FVG
            {"open": 1.0956, "high": 1.0968, "low": 1.0958, "close": 1.0966},  # 8: B2
            {"open": 1.0966, "high": 1.0975, "low": 1.0965, "close": 1.0972},  # 9: B3 + MSS (close > 1.0970)
            {"open": 1.0972, "high": 1.0974, "low": 1.0970, "close": 1.0973},  # 10: ACTIVE CANDLE
        ])

        setup = self.detector.detect_smc_pending(
            df_htf=df_htf,
            df_ltf=df_ltf,
            atr_ltf=0.0010,
            spread=0.0001,
            swing_lookback=2,
            mss_max_bars=3,
            min_rr=1.5
        )

        self.assertIsNotNone(setup)
        self.assertTrue(setup.is_pending)
        self.assertEqual(setup.order_type, "BUY_LIMIT")
        self.assertEqual(setup.direction, "BUY")
        self.assertTrue(setup.has_mss)
        self.assertTrue(setup.has_fvg)
        self.assertEqual(setup.entry_price, 1.09615)
        self.assertEqual(setup.sl, 1.09390)
        self.assertEqual(setup.tp, 1.1000)
        self.assertGreaterEqual(setup.rr, 1.5)

    def test_smc_pending_bearish_valid(self):
        """
        Setup baissier :
        HTF Range : C1 High = 1.1000, Low = 1.0900
        LTF :
          - Swing Low à 1.0975 (lookback 2)
          - Sweep au-dessus de 1.1000 (High = 1.1008)
          - Réintégration close < 1.1000
          - MSS impulsif sous 1.0975
          - FVG baissier créé
          - Ordre SELL_LIMIT au milieu du FVG
        """
        df_htf = pd.DataFrame([
            {"open": 1.0960, "high": 1.0990, "low": 1.0900, "close": 1.0970},
            {"open": 1.0970, "high": 1.1000, "low": 1.0900, "close": 1.0980},  # C1
            {"open": 1.0980, "high": 1.0985, "low": 1.0975, "close": 1.0980},  # Active
        ])

        df_ltf = pd.DataFrame([
            {"open": 1.0985, "high": 1.0990, "low": 1.0980, "close": 1.0985},
            {"open": 1.0985, "high": 1.0988, "low": 1.0980, "close": 1.0982},
            {"open": 1.0982, "high": 1.0985, "low": 1.0975, "close": 1.0980},  # 2: SWING LOW
            {"open": 1.0980, "high": 1.0995, "low": 1.0978, "close": 1.0990},
            {"open": 1.0990, "high": 1.0998, "low": 1.0980, "close": 1.0995},  # 4: CONFIRMATION SWING LOW
            {"open": 1.0995, "high": 1.1008, "low": 1.0990, "close": 1.1004},  # 5: SWEEP HIGH (1.1008 > 1.1000)
            {"open": 1.1004, "high": 1.1004, "low": 1.0985, "close": 1.0990},  # 6: REINTEGRATION (close 1.0990 < 1.1000)
            {"open": 1.0990, "high": 1.0992, "low": 1.0980, "close": 1.0982},  # 7: B1 du FVG (low = 1.0980)
            {"open": 1.0982, "high": 1.0982, "low": 1.0976, "close": 1.0978},  # 8: B2 (close 1.0978 > 1.0975, pas de MSS encore)
            {"open": 1.0978, "high": 1.0974, "low": 1.0965, "close": 1.0968},  # 9: B3 (high = 1.0974 < 1.0980) + MSS (close < 1.0975)
            {"open": 1.0968, "high": 1.0970, "low": 1.0965, "close": 1.0967},  # 10: ACTIVE CANDLE
        ])

        setup = self.detector.detect_smc_pending(
            df_htf=df_htf,
            df_ltf=df_ltf,
            atr_ltf=0.0010,
            spread=0.0001,
            swing_lookback=2,
            mss_max_bars=3,
            min_rr=1.5
        )

        self.assertIsNotNone(setup)
        self.assertTrue(setup.is_pending)
        self.assertEqual(setup.order_type, "SELL_LIMIT")
        self.assertEqual(setup.direction, "SELL")
        self.assertTrue(setup.has_mss)
        self.assertTrue(setup.has_fvg)
        self.assertEqual(setup.entry_price, 1.09770)
        self.assertEqual(setup.sl, 1.10090)
        self.assertEqual(setup.tp, 1.0900)
        self.assertGreaterEqual(setup.rr, 1.5)

    def test_smc_pending_rejected_no_mss(self):
        """Si le prix réintègre mais ne casse jamais le swing précédent dans les 3 bougies, rejet."""
        df_htf = pd.DataFrame([
            {"open": 1.0960, "high": 1.0990, "low": 1.0950, "close": 1.0970},
            {"open": 1.0970, "high": 1.1000, "low": 1.0950, "close": 1.0980},
            {"open": 1.0980, "high": 1.0985, "low": 1.0975, "close": 1.0980},
        ])
        df_ltf = pd.DataFrame([
            {"open": 1.0955, "high": 1.0958, "low": 1.0952, "close": 1.0955},
            {"open": 1.0955, "high": 1.0960, "low": 1.0953, "close": 1.0958},
            {"open": 1.0958, "high": 1.0980, "low": 1.0955, "close": 1.0975},  # High trop haut (1.0980)
            {"open": 1.0975, "high": 1.0962, "low": 1.0952, "close": 1.0955},
            {"open": 1.0955, "high": 1.0958, "low": 1.0948, "close": 1.0952},
            {"open": 1.0952, "high": 1.0952, "low": 1.0940, "close": 1.0945},  # Sweep
            {"open": 1.0945, "high": 1.0955, "low": 1.0944, "close": 1.0954},  # Reintegration
            {"open": 1.0954, "high": 1.0958, "low": 1.0951, "close": 1.0956},
            {"open": 1.0956, "high": 1.0960, "low": 1.0955, "close": 1.0958},  # Toujours sous 1.0980
            {"open": 1.0958, "high": 1.0962, "low": 1.0956, "close": 1.0960},  # Pas de MSS
            {"open": 1.0960, "high": 1.0962, "low": 1.0958, "close": 1.0961},
        ])
        setup = self.detector.detect_smc_pending(
            df_htf=df_htf, df_ltf=df_ltf, atr_ltf=0.0010, spread=0.0001, swing_lookback=2, mss_max_bars=3
        )
        self.assertIsNone(setup)

    def test_smc_pending_rejected_fvg_breached(self):
        """Si une clôture ultérieure traverse la borne opposée du FVG, setup invalidé."""
        df_htf = pd.DataFrame([
            {"open": 1.0960, "high": 1.0990, "low": 1.0950, "close": 1.0970},
            {"open": 1.0970, "high": 1.1000, "low": 1.0950, "close": 1.0980},
            {"open": 1.0980, "high": 1.0985, "low": 1.0975, "close": 1.0980},
        ])
        df_ltf = pd.DataFrame([
            {"open": 1.0955, "high": 1.0958, "low": 1.0952, "close": 1.0955},
            {"open": 1.0955, "high": 1.0960, "low": 1.0953, "close": 1.0958},
            {"open": 1.0958, "high": 1.0970, "low": 1.0955, "close": 1.0965},
            {"open": 1.0965, "high": 1.0962, "low": 1.0952, "close": 1.0955},
            {"open": 1.0955, "high": 1.0958, "low": 1.0948, "close": 1.0952},
            {"open": 1.0952, "high": 1.0952, "low": 1.0940, "close": 1.0945},  # Sweep
            {"open": 1.0945, "high": 1.0955, "low": 1.0944, "close": 1.0954},  # Reintegration
            {"open": 1.0954, "high": 1.0958, "low": 1.0951, "close": 1.0956},  # FVG b1 high 1.0958
            {"open": 1.0956, "high": 1.0968, "low": 1.0958, "close": 1.0966},
            {"open": 1.0966, "high": 1.0975, "low": 1.0965, "close": 1.0972},  # MSS + FVG [1.0958, 1.0965]
            {"open": 1.0972, "high": 1.0972, "low": 1.0950, "close": 1.0950},  # Cassure sous 1.0958 !
            {"open": 1.0950, "high": 1.0952, "low": 1.0948, "close": 1.0950},
        ])
        setup = self.detector.detect_smc_pending(
            df_htf=df_htf, df_ltf=df_ltf, atr_ltf=0.0010, spread=0.0001, swing_lookback=2, mss_max_bars=3
        )
        self.assertIsNone(setup)

    def test_trade_execution_pending_order_request(self):
        """Vérifie la création correcte d'un ordre en attente BUY_LIMIT / SELL_LIMIT."""
        from src.trading.trade_execution import TradeExecutionEngine
        import MetaTrader5 as mt5

        mock_mt5 = MagicMock()
        mock_mt5.get_symbol_info.return_value = {
            "ask": 1.0970,
            "bid": 1.0968,
            "digits": 5,
            "trade_tick_value": 1.0,
            "trade_tick_size": 0.00001,
            "trade_stops_level": 10,
            "point": 0.00001,
        }
        mock_mt5.get_open_positions.return_value = []
        mock_mt5.get_daily_profit.return_value = 0.0

        engine = TradeExecutionEngine(
            mt5_connector=mock_mt5,
            mode="auto",
            min_confidence=50,
            min_reward_risk=1.5
        )

        signal = {
            "symbol": "EURUSD",
            "decision": {
                "action": "BUY",
                "is_pending": True,
                "order_type": "BUY_LIMIT",
                "entry": 1.0960,  # Strictement sous Ask (1.0970)
                "sl": 1.0940,
                "tp": 1.1000,
                "lot": 0.05,
                "confidence": 85,
                "reason": "CRT Universal SMC Pending Limit",
                "expiration_bars": 4,
                "expiration_seconds": 3600,
            },
            "tech_features": {"atr_14": 0.0010},
            "economic_context": {"status": "OK", "event_risk": "LOW"},
        }

        req = engine.prepare_order_request(signal)
        self.assertIsNotNone(req)
        self.assertEqual(req["action"], mt5.TRADE_ACTION_PENDING)
        self.assertEqual(req["type"], mt5.ORDER_TYPE_BUY_LIMIT)
        self.assertEqual(req["price"], 1.0960)
        self.assertEqual(req["sl"], 1.0940)
        self.assertEqual(req["tp"], 1.1000)
        self.assertEqual(req["volume"], 0.05)
        self.assertEqual(req["type_time"], mt5.ORDER_TIME_SPECIFIED)
        self.assertIn("expiration", req)
        self.assertGreater(req["expiration"], 0)

    def test_trade_manager_manage_pending_orders(self):
        """Vérifie l'annulation des pending orders si TP touché avant entrée ou si cible journalière atteinte."""
        from src.trading.trade_manager import TradeManager

        mock_mt5 = MagicMock()
        mock_mt5.cancel_pending_order.return_value = True

        manager = TradeManager(
            mt5_connector=mock_mt5,
            orchestrator=None,
            config={"risk_management": {"daily_profit_target_usd": 50.0}}
        )

        # Scénario 1 : Le TP d'un BUY_LIMIT a été touché avant l'entrée (Bid >= TP)
        mock_mt5.get_pending_orders.return_value = [
            {
                "ticket": 1001,
                "symbol": "EURUSD",
                "type": "BUY",
                "price_open": 1.0960,
                "sl": 1.0940,
                "tp": 1.1000,
                "time_expiration": 0,
            }
        ]
        mock_mt5.get_symbol_info.return_value = {"bid": 1.1005, "ask": 1.1006}
        mock_mt5.get_daily_profit.return_value = 10.0

        cancelled = manager.manage_pending_orders()
        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0]["ticket"], 1001)
        self.assertIn("Cible TP", cancelled[0]["reason"])
        mock_mt5.cancel_pending_order.assert_called_with(1001)

        # Scénario 2 : Cible journalière atteinte
        mock_mt5.get_daily_profit.return_value = 55.0
        mock_mt5.get_symbol_info.return_value = {"bid": 1.0965, "ask": 1.0966}
        cancelled2 = manager.manage_pending_orders()
        self.assertEqual(len(cancelled2), 1)
        self.assertIn("Objectif journalier", cancelled2[0]["reason"])

    def test_trade_execution_simple_market_order_request(self):
        """Vérifie la création correcte d'un ordre simple au marché (TRADE_ACTION_DEAL / exécution directe)."""
        from src.trading.trade_execution import TradeExecutionEngine
        import MetaTrader5 as mt5

        mock_mt5 = MagicMock()
        mock_mt5.get_symbol_info.return_value = {
            "ask": 1.0968,
            "bid": 1.0967,
            "digits": 5,
            "trade_tick_value": 1.0,
            "trade_tick_size": 0.00001,
            "trade_stops_level": 10,
            "point": 0.00001,
        }
        mock_mt5.get_open_positions.return_value = []
        mock_mt5.get_daily_profit.return_value = 0.0

        engine = TradeExecutionEngine(
            mt5_connector=mock_mt5,
            mode="auto",
            min_confidence=50,
            min_reward_risk=1.5
        )

        signal = {
            "symbol": "EURUSD",
            "decision": {
                "action": "BUY",
                "is_pending": False,  # Ordre simple au marché
                "order_type": "MARKET",
                "entry": 1.0968,
                "sl": 1.0940,
                "tp": 1.1010,
                "lot": 0.02,
                "confidence": 80,
                "reason": "CRT Simple Market Order",
            },
            "tech_features": {"atr_14": 0.0010},
            "economic_context": {"status": "OK", "event_risk": "LOW"},
        }

        req = engine.prepare_order_request(signal)
        self.assertIsNotNone(req)
        # Ordre simple direct au marché
        self.assertEqual(req["action"], mt5.TRADE_ACTION_DEAL)
        self.assertEqual(req["type"], mt5.ORDER_TYPE_BUY)
        self.assertEqual(req["price"], 1.0968)
        self.assertEqual(req["sl"], 1.0940)
        self.assertEqual(req["tp"], 1.1010)
        self.assertEqual(req["type_time"], mt5.ORDER_TIME_GTC)
        self.assertNotIn("expiration", req)


if __name__ == "__main__":
    unittest.main()

