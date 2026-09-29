import unittest
import pandas as pd
import numpy as np
from src.core.order_block_detector import OrderBlockDetector


class TestOrderBlockDetector(unittest.TestCase):
    def setUp(self):
        self.detector = OrderBlockDetector(
            impulse_atr_mult=1.6,
            lookback=50,
            retest_tolerance=0.8,
            tp_rr_mult=2.5,
        )

    def test_empty_or_small_df(self):
        self.assertEqual(self.detector.detect(None, "EURUSD", "M15"), [])
        small_df = pd.DataFrame({"close": [1.0] * 10})
        self.assertEqual(self.detector.detect(small_df, "EURUSD", "M15"), [])

    def test_detect_bullish_ob(self):
        np.random.seed(42)
        prices = [1.1000]
        for _ in range(30):
            prices.append(prices[-1] + np.random.normal(0, 0.0002))

        # 1. Bearish candle (the Bullish OB)
        prices.append(prices[-1] - 0.0015)
        # 2. Strong Bullish impulse (displacement)
        prices.append(prices[-1] + 0.0040)
        prices.append(prices[-1] + 0.0020)
        # 3. Retrace back to the OB zone
        prices.append(prices[-2] - 0.0025)
        prices.append(prices[-1] - 0.0020)

        df = pd.DataFrame({
            "open": prices[:-1],
            "close": prices[1:],
            "high": [max(o, c) + 0.0002 for o, c in zip(prices[:-1], prices[1:])],
            "low": [min(o, c) - 0.0002 for o, c in zip(prices[:-1], prices[1:])],
        })

        obs = self.detector.detect(df, "EURUSD", "M15", htf_trend="BULLISH", direction_filter="BUY")
        self.assertGreater(len(obs), 0, "Un Bullish OB aurait dû être détecté")
        ob = obs[0]
        self.assertEqual(ob.direction, "BUY")
        self.assertTrue(ob.is_retested)
        self.assertGreaterEqual(ob.score, 55.0)
        self.assertIsNotNone(ob.entry)
        self.assertIsNotNone(ob.sl)
        self.assertIsNotNone(ob.tp)
        self.assertLess(ob.sl, ob.entry)
        self.assertGreater(ob.tp, ob.entry)
        self.assertGreaterEqual(ob.rr, 2.0)

    def test_detect_bearish_ob(self):
        np.random.seed(123)
        prices = [1.1500]
        for _ in range(30):
            prices.append(prices[-1] + np.random.normal(0, 0.0002))

        # 1. Bullish candle (the Bearish OB)
        prices.append(prices[-1] + 0.0015)
        # 2. Strong Bearish impulse (displacement)
        prices.append(prices[-1] - 0.0040)
        prices.append(prices[-1] - 0.0020)
        # 3. Retrace back to the OB zone
        prices.append(prices[-2] + 0.0025)
        prices.append(prices[-1] + 0.0020)

        df = pd.DataFrame({
            "open": prices[:-1],
            "close": prices[1:],
            "high": [max(o, c) + 0.0002 for o, c in zip(prices[:-1], prices[1:])],
            "low": [min(o, c) - 0.0002 for o, c in zip(prices[:-1], prices[1:])],
        })

        obs = self.detector.detect(df, "EURUSD", "M15", htf_trend="BEARISH", direction_filter="SELL")
        self.assertGreater(len(obs), 0, "Un Bearish OB aurait dû être détecté")
        ob = obs[0]
        self.assertEqual(ob.direction, "SELL")
        self.assertTrue(ob.is_retested)
        self.assertGreaterEqual(ob.score, 55.0)
        self.assertGreater(ob.sl, ob.entry)
        self.assertLess(ob.tp, ob.entry)
        self.assertGreaterEqual(ob.rr, 2.0)

    def test_invalidated_ob_is_filtered(self):
        np.random.seed(42)
        prices = [1.1000]
        for _ in range(30):
            prices.append(prices[-1] + np.random.normal(0, 0.0002))

        # 1. Bearish candle (OB zone)
        prices.append(prices[-1] - 0.0010)
        ob_low = prices[-1] - 0.0002
        # 2. Impulse
        prices.append(prices[-1] + 0.0035)
        # 3. Complete crash way below ob_low (invalidation)
        prices.append(ob_low - 0.0050)
        prices.append(ob_low - 0.0060)

        df = pd.DataFrame({
            "open": prices[:-1],
            "close": prices[1:],
            "high": [max(o, c) + 0.0002 for o, c in zip(prices[:-1], prices[1:])],
            "low": [min(o, c) - 0.0002 for o, c in zip(prices[:-1], prices[1:])],
        })

        obs = self.detector.detect(df, "EURUSD", "M15", direction_filter="BUY")
        self.assertEqual(len(obs), 0, "L'OB invalidé ne doit pas être renvoyé")


if __name__ == "__main__":
    unittest.main()
