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

    def test_detect_swing_points_zero_lookahead_bias(self):
        """Vérifie que les swings ne sont confirmés qu'après clôture des barres de droite (zéro lookahead)."""
        # Pic évident à l'indice 4 (valeur 1.30)
        closes = [1.10, 1.12, 1.15, 1.20, 1.30, 1.22, 1.18, 1.15, 1.12, 1.10]
        df = make_df(closes, spread=0.001)

        # À la bougie 4 (au moment du pic), les barres futures n'existent pas encore
        df_at_peak = df.iloc[:5].copy()
        res_peak = self.engine.detect_swing_points(df_at_peak, window=5)
        # Le pic ne doit PAS encore être confirmé à la bougie du pic
        self.assertTrue(np.isnan(res_peak.iloc[4]['swing_high']))

        # À la bougie 5 (seulement 1 barre de confirmation à droite), toujours pas confirmé
        df_at_1_confirm = df.iloc[:6].copy()
        res_1_confirm = self.engine.detect_swing_points(df_at_1_confirm, window=5)
        self.assertTrue(np.isnan(res_1_confirm.iloc[4]['swing_high']))

        # À la bougie 6 (2 barres de confirmation fermées à droite), le pic à l'indice 4 EST confirmé
        df_at_2_confirm = df.iloc[:7].copy()
        res_2_confirm = self.engine.detect_swing_points(df_at_2_confirm, window=5)
        self.assertFalse(np.isnan(res_2_confirm.iloc[4]['swing_high']))
        # Les deux dernières barres de n'importe quel historique ne peuvent jamais être confirmées à l'avance
        full_res = self.engine.detect_swing_points(df, window=5)
        self.assertTrue(np.isnan(full_res.iloc[-1]['swing_high']))
        self.assertTrue(np.isnan(full_res.iloc[-2]['swing_high']))

    def test_analyze_returns_all_keys(self):
        closes = [1.10 + i * 0.005 for i in range(30)]
        df = make_df(closes)
        res = self.engine.analyze(df)
        for key in ("Phase", "FVG", "BOS_MSS", "Liquidity_Grab"):
            self.assertIn(key, res)

    def test_fvg_closed_bars_only_ignores_forming_bar(self):
        """Si la dernière barre en cours crée un FVG non clôturé, closed_bars_only l'ignore."""
        # 3 barres calmes, puis une 4e barre spike artificiel (en cours)
        closes = [1.10, 1.10, 1.10, 1.25]
        df = make_df(closes)
        # Par défaut avec barre en cours : détecte FVG
        self.assertEqual(self.engine.detect_fvg(df, closed_bars_only=False), "Bullish FVG")
        # En forçant barres fermées uniquement : ignore la 4e barre et retourne None
        self.assertEqual(self.engine.detect_fvg(df, closed_bars_only=True), "None")

    def test_bos_mss_fresh_breakout_only(self):
        """Vérifie qu'un BOS n'est déclenché qu'au moment précis de la cassure et pas sur 10 barres successives."""
        # Pic confirmé à 1.20 à l'indice 4
        closes = [1.10, 1.12, 1.15, 1.18, 1.20, 1.14, 1.12, 1.10, 1.15, 1.18]
        df = make_df(closes)
        # Aucune cassure au début
        self.assertEqual(self.engine.detect_bos_mss(df, fresh_breakout_only=True), "None")

        # Ajout de la bougie de cassure (1.22 > 1.20)
        df_break = make_df(closes + [1.22])
        self.assertEqual(self.engine.detect_bos_mss(df_break, fresh_breakout_only=True), "Bullish BOS/MSS")

        # Bougie suivante qui reste au-dessus (1.23) -> Ce n'est plus une nouvelle cassure (prev_close était déjà > 1.20)
        df_stale = make_df(closes + [1.22, 1.23])
        self.assertEqual(self.engine.detect_bos_mss(df_stale, fresh_breakout_only=True), "None")
        # Mais si fresh_breakout_only est désactivé, elle continue de retourner Bullish BOS/MSS
        self.assertEqual(self.engine.detect_bos_mss(df_stale, fresh_breakout_only=False), "Bullish BOS/MSS")


if __name__ == "__main__":
    unittest.main(verbosity=2)