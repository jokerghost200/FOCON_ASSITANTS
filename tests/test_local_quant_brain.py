"""
Tests unitaires pour LocalQuantBrain (Cerveau Quantitatif Local) :
- Vérification de la calibration statistique (Priorité 4 : un score n'est pas une probabilité brute)
- Détection déterministe des régimes de marché (Priorité 5)
- Principe du 'NO-TRADE as an edge' et journalisation d'invalidation (Priorité 5)
"""
import unittest
from src.ai.local_quant_brain import LocalQuantBrain


class TestLocalQuantBrain(unittest.TestCase):
    def setUp(self):
        self.brain = LocalQuantBrain(min_confidence=65.0, min_rr=1.5)

    def test_calibrate_probability_bounds_and_monotony(self):
        """Vérifie que la probabilité statistique calibrée reste réaliste et bornée."""
        # Score de 0 -> probabilité minimale (0.35)
        p_0 = LocalQuantBrain.calibrate_probability(0.0)
        self.assertAlmostEqual(p_0, 0.35, delta=0.01)

        # Score médian 65 -> ~ 50%
        p_65 = LocalQuantBrain.calibrate_probability(65.0)
        self.assertAlmostEqual(p_65, 0.50, delta=0.02)

        # Score très élevé 95 -> borné à un niveau réaliste pour un système quantitatif (<= 68%)
        p_95 = LocalQuantBrain.calibrate_probability(95.0, market_regime="TRENDING_BULLISH")
        self.assertLessEqual(p_95, 0.68)
        self.assertGreater(p_95, 0.60)

        # Monotonie : un score plus élevé doit donner une probabilité supérieure ou égale
        self.assertGreater(
            LocalQuantBrain.calibrate_probability(80.0),
            LocalQuantBrain.calibrate_probability(60.0)
        )

    def test_expected_value_r_calculation(self):
        """Vérifie le calcul de l'espérance mathématique en unités R : E = P*RR - (1-P)*1."""
        # P = 0.50, RR = 2.0 -> E = 0.5*2 - 0.5*1 = 0.5 R
        ev = LocalQuantBrain.calculate_expected_value_r(win_prob=0.50, reward_risk_ratio=2.0)
        self.assertAlmostEqual(ev, 0.50, places=2)

        # P = 0.30, RR = 1.5 -> E = 0.3*1.5 - 0.7*1 = 0.45 - 0.7 = -0.25 R (espérance négative)
        ev_neg = LocalQuantBrain.calculate_expected_value_r(win_prob=0.30, reward_risk_ratio=1.5)
        self.assertAlmostEqual(ev_neg, -0.25, places=2)

    def test_detect_market_regime(self):
        """Vérifie la détection autonome du régime de marché."""
        # Régime Bullish
        features_bull = {
            "close": 1.1050,
            "ema_50": 1.1000,
            "ema_200": 1.0900,
            "rsi_14": 55.0,
            "adx": 28.0,
            "trend": "BULLISH",
        }
        self.assertEqual(
            LocalQuantBrain.detect_market_regime(features_bull),
            "TRENDING_BULLISH"
        )

        # Régime Bearish
        features_bear = {
            "close": 1.0850,
            "ema_50": 1.0900,
            "ema_200": 1.1000,
            "rsi_14": 45.0,
            "adx": 25.0,
            "trend": "BEARISH",
        }
        self.assertEqual(
            LocalQuantBrain.detect_market_regime(features_bear),
            "TRENDING_BEARISH"
        )

        # Régime Consolidation / Range
        features_range = {
            "close": 1.1000,
            "ema_50": 1.1001,
            "ema_200": 1.1002,
            "rsi_14": 50.0,
            "adx": 12.0,
            "trend": "NEUTRAL",
        }
        self.assertEqual(
            LocalQuantBrain.detect_market_regime(features_range),
            "RANGE_CONSOLIDATION"
        )

    def test_decide_wait_provides_structured_journal_and_invalidation(self):
        """Priorité 5 : Vérifie que le NO-TRADE produit une entrée de journal exploitable et documentée."""
        features = {
            "close": 1.1000,
            "ema_50": 1.1000,
            "ema_200": 1.1000,
            "rsi_14": 50.0,
            "atr_14": 0.0010,
            "trend": "NEUTRAL",
        }
        decision = self.brain.decide("EURUSD", "M15", features)
        self.assertEqual(decision["action"], "WAIT")
        self.assertIn("invalidation_reason", decision)
        self.assertIn("journal_record", decision)
        self.assertIn("statistical_win_prob", decision)
        self.assertIn("technical_score", decision)

        rec = decision["journal_record"]
        self.assertEqual(rec["action"], "WAIT")
        self.assertEqual(rec["symbol"], "EURUSD")
        self.assertIsNotNone(rec["invalidation_reason"])

    def test_decide_buy_provides_technical_score_and_calibrated_probability(self):
        """Priorité 4 & 5 : Vérifie la séparation entre score technique et probabilité calibrée sur BUY."""
        indicators = {
            "close": 100.0,
            "rsi_14": 50.0,
            "ema_50": 98.0,
            "ema_200": 90.0,
            "macd": 0.5,
            "macd_signal": 0.2,
            "macd_hist": 0.3,
            "adx": 30.0,
            "atr_14": 1.5,
            "trend": "BULLISH",
        }
        boomerang_data = {"decision": "VALID", "direction": "BUY", "score": 85.0}
        decision = self.brain.decide(
            "BTCUSD", "M15", indicators,
            boomerang_data=boomerang_data,
            htf_trend="BULLISH"
        )

        self.assertEqual(decision["action"], "BUY")
        self.assertIn("technical_score", decision)
        self.assertIn("statistical_win_prob", decision)
        self.assertIn("expected_value_r", decision)
        self.assertIn("market_regime", decision)
        self.assertIn("journal_record", decision)

        # La probabilité calibrée doit être dans un intervalle réaliste [0.50, 0.68]
        self.assertGreaterEqual(decision["statistical_win_prob"], 0.50)
        self.assertLessEqual(decision["statistical_win_prob"], 0.68)
        # L'espérance mathématique doit être positive
        self.assertGreater(decision["expected_value_r"], 0.0)


if __name__ == "__main__":
    unittest.main()
