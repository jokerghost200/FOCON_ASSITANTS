"""
Tests unitaires pour TechnicalIndicators (données synthétiques, sans MT5).
Lancer avec :  python -m pytest tests/ -v  (ou python tests/test_technical_indicators.py)
"""
import os
import sys
import unittest

import pandas as pd
import numpy as np

# Racine du projet (parent de tests/) : permet `from src...` où que soit lancé pytest.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.technical_indicators import TechnicalIndicators


def make_synthetic_df(n=300, trend=0.001):
    """Génère un DataFrame OHLCV synthétique avec une dérive légère."""
    rng = np.random.default_rng(42)
    close = 1.1000 + np.cumsum(rng.normal(trend, 0.0008, n))
    high = close + abs(rng.normal(0, 0.0004, n))
    low = close - abs(rng.normal(0, 0.0004, n))
    open_ = close - rng.normal(0, 0.0003, n)
    index = pd.date_range("2026-01-01", periods=n, freq="5min")
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "tick_volume": rng.integers(1, 500, n), "spread": 1,
    }, index=index)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df


class TestTechnicalIndicators(unittest.TestCase):

    def setUp(self):
        self.tech = TechnicalIndicators()
        self.df = make_synthetic_df()

    def test_add_all_indicators_columns(self):
        out = self.tech.add_all_indicators(self.df)
        for col in ["EMA_50", "EMA_200", "MACD", "MACD_signal", "MACD_hist",
                    "RSI_14", "ATR_14", "last_swing_high", "last_swing_low",
                    "bullish_bias", "bearish_bias"]:
            self.assertIn(col, out.columns, f"Column {col} missing")

    def test_atr_positive(self):
        out = self.tech.add_all_indicators(self.df)
        self.assertGreater(out["ATR_14"].dropna().iloc[-1], 0)

    def test_rsi_bounds(self):
        out = self.tech.add_all_indicators(self.df)
        rsi = out["RSI_14"].dropna().iloc[-1]
        self.assertTrue(0 <= rsi <= 100, f"RSI out of bounds: {rsi}")

    def test_extract_latest_features(self):
        out = self.tech.add_all_indicators(self.df)
        feats = self.tech.extract_latest_features(out)
        self.assertIn("trend", feats)
        self.assertIn("rsi_14", feats)
        # Aucune valeur ne doit rester en NaN
        for k, v in feats.items():
            if v is not None and not isinstance(v, str):
                self.assertFalse(np.isnan(v), f"NaN value for {k}")

    def test_insufficient_data(self):
        small = self.df.head(10)
        out = self.tech.add_all_indicators(small)
        self.assertNotIn("EMA_50", out.columns)  # pas assez de données


if __name__ == "__main__":
    unittest.main(verbosity=2)