"""
Tests unitaires pour le système de Conscience Évolutive, Mémoire et Auto-Critique.
"""
import unittest
import os
import shutil
from src.learning.trade_memory import TradeMemoryEngine
from src.learning.reflective_critic import ReflectiveCritic
from src.learning.adaptive_weights import AdaptiveWeightsEngine, DEFAULT_WEIGHTS, WEIGHT_BOUNDS
from src.ai.local_quant_brain import LocalQuantBrain


class TestLearningSystem(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_data_learning"
        os.makedirs(self.test_dir, exist_ok=True)
        self.weights_path = os.path.join(self.test_dir, "adaptive_weights.json")
        self.memory_path = os.path.join(self.test_dir, "trade_memory.json")

        self.weights_engine = AdaptiveWeightsEngine(file_path=self.weights_path, learning_rate=0.5)
        self.critic = ReflectiveCritic(ai_router=None)  # deterministic fallback
        self.trade_memory = TradeMemoryEngine(
            file_path=self.memory_path,
            critic=self.critic,
            weights_engine=self.weights_engine,
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_record_entry_and_update_excursion(self):
        """Vérifie l'enregistrement du trade et le suivi dynamique de ses excursions (MFE/MAE)."""
        snapshot = {
            "rsi_14": 48.5,
            "ema_dist_atr": 0.8,
            "htf_trend": "BULLISH",
            "has_ob": True,
            "ob_score": 75,
            "reason": "Retest Bullish Order Block",
        }
        rec = self.trade_memory.record_entry(
            ticket=1001,
            symbol="BTCUSD",
            action="BUY",
            lot=0.05,
            price=60000.0,
            sl=59500.0,
            tp=61500.0,
            snapshot=snapshot,
        )
        self.assertEqual(rec["ticket"], 1001)
        self.assertEqual(rec["status"], "OPEN")

        # Excursion positive puis négative
        self.trade_memory.update_excursion(1001, 12.50)
        self.trade_memory.update_excursion(1001, -3.20)
        self.trade_memory.update_excursion(1001, 8.00)

        trade = self.trade_memory.trades[1001]
        self.assertEqual(trade["max_profit_usd"], 12.50)
        self.assertEqual(trade["max_drawdown_usd"], -3.20)
        self.assertEqual(trade["last_profit_usd"], 8.00)

    def test_record_exit_triggers_critique_and_lesson(self):
        """Vérifie que la clôture génère l'auto-critique et une leçon actionnable."""
        self.trade_memory.record_entry(
            ticket=2001,
            symbol="ETHUSD",
            action="BUY",
            lot=0.1,
            price=2500.0,
            sl=2450.0,
            tp=2650.0,
            snapshot={"has_ob": True, "ob_score": 80, "rsi_14": 50.0, "htf_trend": "BULLISH"},
        )
        self.trade_memory.update_excursion(2001, 15.00)
        
        closed = self.trade_memory.record_exit(
            ticket=2001,
            close_price=2600.0,
            final_profit_usd=14.50,
            exit_reason="DOLLAR_LOCK",
        )
        self.assertEqual(closed["status"], "CLOSED")
        self.assertEqual(closed["final_profit_usd"], 14.50)
        self.assertIsNotNone(closed["critique"])
        self.assertIsNotNone(closed["lesson"])
        self.assertIn("WIN", closed["tags"])
        self.assertIn("TARGET_HIT", closed["tags"])

    def test_critic_diagnoses_profit_giveback_greed(self):
        """Vérifie que l'auto-critique détecte la gourmandise si un gros profit a été rendu."""
        trade_data = {
            "ticket": 3001,
            "symbol": "SOLUSD",
            "action": "BUY",
            "entry_price": 150.0,
            "close_price": 149.0,
            "final_profit_usd": -2.00,
            "max_profit_usd": 18.00,  # Est monté à +18$ avant de finir à -2$ !
            "max_drawdown_usd": -3.00,
            "exit_reason": "SL_HIT",
            "snapshot": {"rsi_14": 52.0, "htf_trend": "BULLISH"},
        }
        res = self.critic.critique_trade(trade_data, use_ai=False)
        self.assertIn("PROFIT_GIVEBACK", res["tags"])
        self.assertIn("Gourmandise", res["critique"])
        self.assertIn("verrouiller les gains", res["lesson"])

    def test_critic_diagnoses_counter_htf_error(self):
        """Vérifie que l'auto-critique sanctionne un trade pris contre la tendance de fond HTF."""
        trade_data = {
            "ticket": 3002,
            "symbol": "EURUSD",
            "action": "BUY",
            "entry_price": 1.1000,
            "close_price": 1.0950,
            "final_profit_usd": -15.00,
            "max_profit_usd": 0.05,
            "max_drawdown_usd": -15.00,
            "exit_reason": "SL_HIT",
            "snapshot": {"rsi_14": 55.0, "htf_trend": "BEARISH"},
        }
        res = self.critic.critique_trade(trade_data, use_ai=False)
        self.assertIn("COUNTER_HTF", res["tags"])
        self.assertIn("contre-courant", res["critique"])
        self.assertIn("Interdire formellement", res["lesson"])

    def test_adaptive_weights_recalibration_boosts_winning_setup(self):
        """Vérifie que des trades gagnants basés sur Order Block augmentent son poids."""
        trades = []
        # Générer 5 trades gagnants basés sur OB
        for i in range(5):
            trades.append({
                "ticket": 4000 + i,
                "symbol": "BTCUSD",
                "final_profit_usd": 12.0,
                "snapshot": {"has_ob": True, "ob_score": 75, "rsi_14": 50},
            })
        initial_ob_w = self.weights_engine.weights["ob_weight"]
        res = self.weights_engine.recalibrate(trades)
        new_ob_w = self.weights_engine.weights["ob_weight"]

        self.assertEqual(res["status"], "RECALIBRATED")
        self.assertGreater(new_ob_w, initial_ob_w)
        self.assertLessEqual(new_ob_w, WEIGHT_BOUNDS["ob_weight"][1])

    def test_local_quant_brain_respects_adaptive_weights(self):
        """Vérifie que le Cerveau Quantitatif Local utilise les poids dynamiques calibrés."""
        # Définir un poids Boomerang élevé
        self.weights_engine.weights["boomerang_weight"] = 45.0
        brain = LocalQuantBrain(min_confidence=60.0, weights_engine=self.weights_engine)

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
        }
        boomerang_data = {"decision": "VALID", "direction": "BUY", "score": 80.0}
        decision = brain.decide("ETHUSD", "M15", indicators, boomerang_data=boomerang_data, htf_trend="BULLISH")
        
        self.assertEqual(decision["action"], "BUY")
        self.assertIn("pds 45", decision["reason"])


if __name__ == "__main__":
    unittest.main()
