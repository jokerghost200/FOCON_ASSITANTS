"""
Tests unitaires pour AMDEngine (Smart Money Concepts, données synthétiques).
Lancer avec :  python -m pytest tests/ -v  (ou python tests/test_amd_detector.py)
"""
import os
import sys
import unittest

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.amd_detector import AMDEngine


def make_df(close_values, spread=0.001):
    """Construit un DataFrame à partir d'une série de closes."""
    closes = np.asarray(close_values, dtype=float)
    highs = closes + spread
    lows = closes - spread
    n = len(closes)
    index = pd.date_range("2026-01-01", periods=n, freq="5min")
    return pd.DataFrame({
        "open": closes, "high": highs, "low": lows, "close": closes,
        "volume": 100, "spread": 1,
    }, index=index)


class TestAMDEngine(unittest.TestCase):

    def setUp(self):
        self.engine = AMDEngine(lookback_period=20)

    def test_bullish_fvg(self):
        # Chandelier écarté vers le haut: low(c3) > high(c1)
        closes = [1.10, 1.10, 1.10, 1.18, 1.19, 1.20]
        df = make_df(closes)
        self.assertEqual(self.engine.detect_fvg(df), "Bullish FVG")

    def test_no_fvg_in_flat(self):
        closes = [1.10] * 20
        df = make_df(closes)
        self.assertEqual(self.engine.detect_fvg(df), "None")

    def test_phase_is_accumulation_when_range_tight(self):
        closes = [1.10 + 0.001 * np.sin(i) for i in range(20)]
        df = make_df(closes)
        phase = self.engine.detect_amd_phase(df)
        self.assertIn(phase, ("Accumulation", "Consolidation"))

    def test_analyze_returns_all_keys(self):
        closes = [1.10 + i * 0.005 for i in range(30)]
        df = make_df(closes)
        res = self.engine.analyze(df)
        for key in ("Phase", "FVG", "BOS_MSS", "Liquidity_Grab"):
            self.assertIn(key, res)


if __name__ == "__main__":
    unittest.main(verbosity=2)