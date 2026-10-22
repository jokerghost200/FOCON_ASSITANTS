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


if __name__ == "__main__":
    unittest.main()
