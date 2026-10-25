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
        # Métadonnées de calibration explicite
        self.assertFalse(decision["is_calibrated"])
        self.assertEqual(decision["probability_status"], "UNCALIBRATED_HEURISTIC")
        self.assertIn("normalized_confluence_pct", decision)

    def test_decide_fail_closed_missing_or_nan_technical_features(self):
        """Phase 1 : Vérifie que des indicateurs manquants, NaN ou Inf provoquent un WAIT immédiat."""
        valid_features = {
            "close": 1.1000, "ema_50": 1.0990, "ema_200": 1.0950,
            "rsi_14": 52.0, "atr_14": 0.0012, "trend": "BULLISH",
        }
        # 1. RSI manquant
        f_no_rsi = dict(valid_features)
        del f_no_rsi["rsi_14"]
        d1 = self.brain.decide("EURUSD", "M15", f_no_rsi)
        self.assertEqual(d1["action"], "WAIT")
        self.assertEqual(d1["market_regime"], "DEGRADED_DATA")
        self.assertIn("Fail-Closed", d1["invalidation_reason"])

        # 2. EMA_50 est NaN
        f_nan_ema = dict(valid_features)
        f_nan_ema["ema_50"] = float("nan")
        d2 = self.brain.decide("EURUSD", "M15", f_nan_ema)
        self.assertEqual(d2["action"], "WAIT")
        self.assertEqual(d2["market_regime"], "DEGRADED_DATA")

        # 3. ATR_14 est nul ou négatif
        f_zero_atr = dict(valid_features)
        f_zero_atr["atr_14"] = 0.0
        d3 = self.brain.decide("EURUSD", "M15", f_zero_atr)
        self.assertEqual(d3["action"], "WAIT")
        self.assertEqual(d3["market_regime"], "DEGRADED_DATA")

        # 4. Close est manquant ou <= 0
        f_bad_close = dict(valid_features)
        f_bad_close["close"] = -1.0
        d4 = self.brain.decide("EURUSD", "M15", f_bad_close)
        self.assertEqual(d4["action"], "WAIT")
        self.assertEqual(d4["market_regime"], "DEGRADED_DATA")

    def test_decide_fail_closed_invalid_sl_tp_geometry(self):
        """Phase 1 : Vérifie qu'un SL ou TP placé du mauvais côté de l'entrée est refusé (WAIT)."""
        indicators = {
            "close": 100.0, "rsi_14": 50.0, "ema_50": 98.0, "ema_200": 90.0,
            "macd": 0.5, "macd_signal": 0.2, "macd_hist": 0.3, "adx": 30.0, "atr_14": 1.5,
            "trend": "BULLISH",
        }
        boomerang_data = {"decision": "VALID", "direction": "BUY", "score": 85.0}

        # Mocking or simulating an invalid SL via monkey-patching min_sl_dist logic or custom OB
        class BadOB:
            direction = "BUY"
            is_retested = True
            score = 80
            ob_type = "BULLISH_OB"
            sl = 105.0  # SL au-dessus de l'entrée pour un BUY -> INVALIDE !

        d_bad_geom = self.brain.decide(
            "BTCUSD", "M15", indicators,
            boomerang_data=boomerang_data,
            order_blocks=[BadOB()],
            htf_trend="BULLISH",
        )
        # Doit être rejeté en WAIT pour géométrie invalide
        self.assertEqual(d_bad_geom["action"], "WAIT")
        self.assertIn("Géométrie", d_bad_geom["invalidation_reason"])

    def test_decide_fail_closed_insufficient_reward_risk_ratio(self):
        """Phase 1 : Vérifie qu'un ratio R:R inférieur à min_rr conduit à WAIT."""
        strict_brain = LocalQuantBrain(min_confidence=60.0, min_rr=3.0)  # Exige un R:R >= 3.0
        indicators = {
            "close": 100.0, "rsi_14": 50.0, "ema_50": 98.0, "ema_200": 90.0,
            "macd": 0.5, "macd_signal": 0.2, "macd_hist": 0.3, "adx": 30.0, "atr_14": 1.5,
            "trend": "BULLISH",
        }
        boomerang_data = {"decision": "VALID", "direction": "BUY", "score": 85.0}

        # Par défaut, le setup BUY propose un RR de ~2.0
        res = strict_brain.decide(
            "BTCUSD", "M15", indicators,
            boomerang_data=boomerang_data,
            htf_trend="BULLISH",
        )
        self.assertEqual(res["action"], "WAIT")
        self.assertIn("Ratio R:R insuffisant", res["invalidation_reason"])

    def test_drift_detector_quarantine_blocks_el_estratega_and_decision_engine(self):
        """Phase 1 & 4 : Vérifie que le DriftDetector en quarantaine critique bloque l'exécution dans El Estratega."""
        from src.cerebros.el_estratega import ElEstratega
        from src.learning.drift_detector import PerformanceDriftDetector

        drift_det = PerformanceDriftDetector(window_size=5, min_trades_to_evaluate=3)
        drift_det.is_quarantined = True  # Dérive critique active

        estratega = ElEstratega(drift_detector=drift_det)
        tech_features = {
            "close": 1.1000, "ema_50": 1.0990, "ema_200": 1.0950,
            "rsi_14": 50.0, "atr_14": 0.0010, "trend": "BULLISH",
        }
        res = estratega.decidir(
            symbol="EURUSD", timeframe="M15", tech_features=tech_features,
            boomerang_data={"decision": "VALID", "direction": "BUY", "score": 90},
            htf_trend="BULLISH",
        )
        self.assertEqual(res["action"], "WAIT")
        self.assertTrue(res.get("is_quarantined", False))
        self.assertIn("QUARANTAINE CRITIQUE", res["reason"])

    def test_adaptive_weights_rollback_and_history_logging(self):
        """Phase 1 & 3 : Vérifie la réversibilité (rollback) et le logging détaillé d'AdaptiveWeightsEngine."""
        import tempfile, os
        from src.learning.adaptive_weights import AdaptiveWeightsEngine
        with tempfile.TemporaryDirectory() as tmp_dir:
            fpath = os.path.join(tmp_dir, "weights_test.json")
            engine = AdaptiveWeightsEngine(file_path=str(fpath), learning_rate=2.0)
            orig_ob = engine.weights["ob_weight"]

            # Modifier le poids
            engine.record_trade_outcome("ORDER_BLOCK", is_win=True, pnl=50.0, result_r=2.5, exit_reason="TP_HIT")
            new_ob = engine.weights["ob_weight"]
            self.assertGreater(new_ob, orig_ob)
            self.assertEqual(len(engine.history_log), 1)
            self.assertEqual(engine.history_log[0]["result_r"], 2.5)

            # Rollback
            restored = engine.rollback()
            self.assertEqual(restored["ob_weight"], orig_ob)


if __name__ == "__main__":
    unittest.main()

