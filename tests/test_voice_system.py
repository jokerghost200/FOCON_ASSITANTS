import unittest
from unittest.mock import MagicMock, patch

from src.core.voice_manager import VoiceManager
from src.core.voice_commands import VoiceCommandDispatcher
from src.learning.market_adapt_sentinel import MarketAdaptationSentinel


class TestVoiceSystem(unittest.TestCase):
    @patch("src.core.voice_manager.QTextToSpeech")
    @patch("src.core.voice_manager.QMediaCaptureSession")
    @patch("src.core.voice_manager.QAudioInput")
    @patch("src.core.voice_manager.QMediaRecorder")
    def test_voice_parameters_tuning(self, mock_rec, mock_inp, mock_sess, mock_tts):
        vm = VoiceManager()
        
        # Test rate setting & clamping
        vm.set_rate(0.5)
        self.assertEqual(vm.rate, 0.5)
        vm.set_rate(2.5)  # should clamp to 1.0
        self.assertEqual(vm.rate, 1.0)
        vm.set_rate(-2.0)  # should clamp to -1.0
        self.assertEqual(vm.rate, -1.0)

        # Test pitch setting & clamping
        vm.set_pitch(0.2)
        self.assertEqual(vm.pitch, 0.2)
        vm.set_pitch(1.5)
        self.assertEqual(vm.pitch, 1.0)

        # Test volume setting & clamping
        vm.set_volume(0.8)
        self.assertEqual(vm.volume, 0.8)
        vm.set_volume(1.5)
        self.assertEqual(vm.volume, 1.0)
        vm.set_volume(-0.5)
        self.assertEqual(vm.volume, 0.0)

        # Test load_config & get_config
        cfg = {
            "rate": 0.3,
            "pitch": -0.1,
            "volume": 0.75,
            "auto_adapt_on_danger": True,
            "voice_announcements": False,
        }
        vm.load_config(cfg)
        self.assertEqual(vm.rate, 0.3)
        self.assertEqual(vm.pitch, -0.1)
        self.assertEqual(vm.volume, 0.75)
        self.assertTrue(vm.auto_adapt_on_danger)
        self.assertFalse(vm.voice_announcements)

        exported = vm.get_config()
        self.assertEqual(exported["rate"], 0.3)
        self.assertEqual(exported["volume"], 0.75)
        self.assertTrue(exported["auto_adapt_on_danger"])

    def test_voice_command_autopilot_toggle(self):
        fake_app = MagicMock()
        fake_app.is_autopilot_scan = False

        # Test "active autopilot" (user's exact sentence)
        res_exact = VoiceCommandDispatcher.dispatch("active autopilot", fake_app)
        self.assertIsNotNone(res_exact)
        self.assertEqual(res_exact["action"], "enable_autopilot")

        # Test "Active l'autopilot s'il te plaît"
        fake_app.is_autopilot_scan = False
        res = VoiceCommandDispatcher.dispatch("Active l'autopilot s'il te plaît", fake_app)
        self.assertIsNotNone(res)
        self.assertEqual(res["action"], "enable_autopilot")
        fake_app.autopilot_btn.setChecked.assert_called_with(True)

        # Test "autopilot on"
        fake_app.is_autopilot_scan = False
        res_on = VoiceCommandDispatcher.dispatch("autopilot on", fake_app)
        self.assertIsNotNone(res_on)
        self.assertEqual(res_on["action"], "enable_autopilot")

        # Test "Arrête l'autopilot immédiatement"
        fake_app.is_autopilot_scan = True
        res_stop = VoiceCommandDispatcher.dispatch("Arrête l'autopilot immédiatement", fake_app)
        self.assertIsNotNone(res_stop)
        self.assertEqual(res_stop["action"], "disable_autopilot")
        fake_app.autopilot_btn.setChecked.assert_called_with(False)

        # Test "desactive autopilot"
        fake_app.is_autopilot_scan = True
        res_desact = VoiceCommandDispatcher.dispatch("desactive autopilot", fake_app)
        self.assertIsNotNone(res_desact)
        self.assertEqual(res_desact["action"], "disable_autopilot")

        # Test "autopilot off"
        fake_app.is_autopilot_scan = True
        res_off = VoiceCommandDispatcher.dispatch("autopilot off", fake_app)
        self.assertIsNotNone(res_off)
        self.assertEqual(res_off["action"], "disable_autopilot")

    def test_voice_command_account_and_positions_query(self):
        fake_app = MagicMock()
        fake_app.orchestrator.mt5.connected = True
        fake_app.orchestrator.mt5.get_account_info.return_value = {
            "balance": 150.0,
            "equity": 165.5,
            "profit": 15.5
        }
        fake_app.orchestrator.mt5.get_open_positions.return_value = [
            {"symbol": "EURUSD", "type": "BUY", "volume": 0.01, "profit": 10.5},
            {"symbol": "GBPUSD", "type": "SELL", "volume": 0.01, "profit": 5.0}
        ]

        # Balance query
        res_acc = VoiceCommandDispatcher.dispatch("Quel est mon solde ?", fake_app)
        self.assertIsNotNone(res_acc)
        self.assertEqual(res_acc["action"], "account_info")
        self.assertIn("150.00", res_acc["reply"])

        # Positions query
        res_pos = VoiceCommandDispatcher.dispatch("Quelles sont mes positions ouvertes ?", fake_app)
        self.assertIsNotNone(res_pos)
        self.assertEqual(res_pos["action"], "open_positions")
        self.assertIn("2 position(s)", res_pos["reply"])
        self.assertIn("EURUSD", res_pos["reply"])

    def test_voice_command_mute_unmute(self):
        fake_app = MagicMock()
        fake_app.voice_manager.is_muted = False

        res_mute = VoiceCommandDispatcher.dispatch("Coupe la voix", fake_app)
        self.assertIsNotNone(res_mute)
        self.assertEqual(res_mute["action"], "mute_voice")
        self.assertTrue(fake_app.voice_manager.is_muted)

        res_unmute = VoiceCommandDispatcher.dispatch("Remets la voix s'il te plaît", fake_app)
        self.assertIsNotNone(res_unmute)
        self.assertEqual(res_unmute["action"], "unmute_voice")
        self.assertFalse(fake_app.voice_manager.is_muted)

    def test_voice_command_risk_profile_change(self):
        fake_app = MagicMock()
        fake_app.risk_profile_combo.findData.return_value = 1

        res = VoiceCommandDispatcher.dispatch("Passe en mode prudent", fake_app)
        self.assertIsNotNone(res)
        self.assertEqual(res["action"], "set_risk_profile")
        self.assertEqual(res["value"], "CONSERVATIVE")
        fake_app.risk_profile_combo.setCurrentIndex.assert_called_with(1)

        res_dyn = VoiceCommandDispatcher.dispatch("Passe en profil dynamique", fake_app)
        self.assertIsNotNone(res_dyn)
        self.assertEqual(res_dyn["action"], "set_risk_profile")
        self.assertEqual(res_dyn["value"], "DYNAMIC")

        # Test Sniper Mode activation
        res_sniper = VoiceCommandDispatcher.dispatch("Active le mode sniper", fake_app)
        self.assertIsNotNone(res_sniper)
        self.assertEqual(res_sniper["action"], "set_risk_profile")
        self.assertEqual(res_sniper["value"], "SNIPER")
        self.assertIn("Sniper", res_sniper["reply"])

        # Test Sniper Mode deactivation
        res_sniper_off = VoiceCommandDispatcher.dispatch("Désactive le mode sniper", fake_app)
        self.assertIsNotNone(res_sniper_off)
        self.assertEqual(res_sniper_off["action"], "set_risk_profile")
        self.assertEqual(res_sniper_off["value"], "BALANCED")

    def test_voice_command_quick_tp_targets(self):
        fake_app = MagicMock()

        # 4$, 5$, 10$, 12$, 15$
        for val in [4, 5, 10, 12, 15]:
            res = VoiceCommandDispatcher.dispatch(f"Objectif {val} dollars", fake_app)
            self.assertIsNotNone(res)
            self.assertEqual(res["action"], "set_quick_tp")
            self.assertEqual(res["value"], float(val))
            fake_app.set_quick_tp_target.assert_called_with(float(val))

        # Test "Sortie à 15 dollars"
        res_exit = VoiceCommandDispatcher.dispatch("Sortie à 15 dollars", fake_app)
        self.assertIsNotNone(res_exit)
        self.assertEqual(res_exit["action"], "set_quick_tp")
        self.assertEqual(res_exit["value"], 15.0)

        # Test "Prends 4 dollars"
        res_take = VoiceCommandDispatcher.dispatch("Prends 4 dollars", fake_app)
        self.assertIsNotNone(res_take)
        self.assertEqual(res_take["action"], "set_quick_tp")
        self.assertEqual(res_take["value"], 4.0)

        # Test Trailing mode
        res_trailing = VoiceCommandDispatcher.dispatch("Passe en mode trailing", fake_app)
        self.assertIsNotNone(res_trailing)
        self.assertEqual(res_trailing["action"], "set_quick_tp")
        self.assertEqual(res_trailing["value"], 0.0)

    def test_voice_command_direction_and_emergency_close(self):
        fake_app = MagicMock()
        fake_app.quick_direction_combo.findData.return_value = 0

        res_dir = VoiceCommandDispatcher.dispatch("Achats uniquement pour aujourd'hui", fake_app)
        self.assertIsNotNone(res_dir)
        self.assertEqual(res_dir["action"], "set_direction")
        self.assertEqual(res_dir["value"], "BUY_ONLY")

        # Emergency close
        fake_app.orchestrator.mt5.connected = True
        fake_app.orchestrator.mt5.get_open_positions.return_value = [
            {"ticket": 101, "symbol": "EURUSD", "type": "BUY", "volume": 0.01, "price_current": 1.0850}
        ]
        fake_app.orchestrator.mt5.close_position.return_value = True

        res_close = VoiceCommandDispatcher.dispatch("Ferme toutes les positions d'urgence", fake_app)
        self.assertIsNotNone(res_close)
        self.assertEqual(res_close["action"], "close_all")
        self.assertIn("1 position", res_close["reply"])

    def test_market_adaptation_sentinel_micro_capital(self):
        fake_orchestrator = MagicMock()
        fake_orchestrator.mt5.connected = True
        fake_orchestrator.mt5.get_account_info.return_value = {
            "balance": 50.0,
            "equity": 50.0,
        }

        config = {
            "risk_management": {
                "default_risk_percent": 0.5,
                "max_open_positions": 3,
                "min_confidence": 60.0,
            },
            "trading": {
                "allowed_market_types": ["Forex", "Crypto"],
            }
        }

        eval_res = MarketAdaptationSentinel.evaluate_market_and_account(fake_orchestrator, config)
        self.assertTrue(eval_res["need_adaptation"])
        keys_changed = [c["key"] for c in eval_res["changes"]]
        self.assertIn("allowed_market_types", keys_changed)
        self.assertIn("max_open_positions", keys_changed)

    def test_market_adaptation_sentinel_loss_streak(self):
        fake_orchestrator = MagicMock()
        fake_orchestrator.mt5.connected = True
        fake_orchestrator.mt5.get_account_info.return_value = {
            "balance": 1000.0,
            "equity": 980.0,
        }

        config = {
            "risk_management": {
                "default_risk_percent": 1.0,
                "max_open_positions": 2,
                "min_confidence": 60.0,
            },
            "trading": {
                "allowed_market_types": ["Forex"],
            }
        }

        # Simuler 2 pertes consécutives récentes
        recent_trades = [
            {"ticket": 1, "profit": -15.0},
            {"ticket": 2, "profit": -12.0},
        ]

        eval_res = MarketAdaptationSentinel.evaluate_market_and_account(
            fake_orchestrator, config, recent_trades=recent_trades
        )
        self.assertTrue(eval_res["need_adaptation"])
        self.assertEqual(eval_res["consecutive_losses"], 2)
        keys_changed = [c["key"] for c in eval_res["changes"]]
        self.assertIn("default_risk_percent", keys_changed)
        self.assertIn("min_confidence", keys_changed)


if __name__ == "__main__":
    unittest.main()
